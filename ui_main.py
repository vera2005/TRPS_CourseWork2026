"""Главное окно PyQt6, видеозахват, калибровка и запись MP4."""

import math
import threading
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pose_processor import PoseProcessor
from rep_counter import RepCounter
from settings_manager import SettingsManager
from stats_logger import ERROR_TITLES, StatsLogger


@lru_cache(maxsize=8)
def load_font(size: int) -> ImageFont.ImageFont:
    """Загружает шрифт с кириллицей для текста на видеокадре."""
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )

    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)

    return ImageFont.load_default()


def draw_overlay(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Наносит UTF-8 текст через Pillow."""
    if not lines:
        return

    image = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(21)

    panel_height = 18 + len(lines) * 29
    draw.rectangle(
        (8, 8, 430, panel_height),
        fill=(0, 0, 0, 165),
    )

    for index, (text, color) in enumerate(lines):
        draw.text(
            (18, 13 + index * 29),
            text,
            font=font,
            fill=color,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    frame[:] = cv2.cvtColor(
        np.asarray(image),
        cv2.COLOR_RGB2BGR,
    )


class VideoRecorder:
    """Пишет обработанные кадры в MP4 кодеком H.264."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.container = av.open(str(path), mode="w")

        safe_fps = (
            max(1, int(round(fps)))
            if math.isfinite(fps) and fps > 0
            else 25
        )

        self.stream = self.container.add_stream(
            "libx264",
            rate=safe_fps,
        )
        self.stream.width = width - width % 2
        self.stream.height = height - height % 2
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {
            "preset": "veryfast",
            "crf": "23",
        }

    def write(self, frame_bgr: np.ndarray) -> None:
        """Конвертирует OpenCV-кадр в RGB и кодирует его."""
        height = self.stream.height
        width = self.stream.width

        if frame_bgr.shape[:2] != (height, width):
            frame_bgr = cv2.resize(
                frame_bgr,
                (width, height),
            )

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        video_frame = av.VideoFrame.from_ndarray(
            rgb,
            format="rgb24",
        )

        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self) -> None:
        """Сбрасывает буфер кодека и завершает MP4-контейнер."""
        if self.container is None:
            return

        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None


class VideoLabel(QLabel):
    """QLabel с преобразованием клика в координаты исходного кадра."""

    video_clicked = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__(
            "Нажмите «Старт», чтобы включить камеру"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            "background-color: #17191c; color: #d8d8d8;"
        )

        self.calibration_enabled = False
        self._frame_size: tuple[int, int] | None = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Отображает BGR-кадр с сохранением пропорций."""
        height, width = frame_bgr.shape[:2]
        self._frame_size = (width, height)

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )
        image = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:
        """Передаёт клик только внутри видеообласти."""
        pixmap = self.pixmap()

        if (
            not self.calibration_enabled
            or pixmap is None
            or self._frame_size is None
        ):
            super().mousePressEvent(event)
            return

        x_offset = (
            self.width() - pixmap.width()
        ) / 2
        y_offset = (
            self.height() - pixmap.height()
        ) / 2

        position = event.position()
        x = position.x() - x_offset
        y = position.y() - y_offset

        if (
            0 <= x < pixmap.width()
            and 0 <= y < pixmap.height()
        ):
            original_width, original_height = (
                self._frame_size
            )
            frame_x = int(
                x * original_width / pixmap.width()
            )
            frame_y = int(
                y * original_height / pixmap.height()
            )
            self.video_clicked.emit(frame_x, frame_y)

        super().mousePressEvent(event)


class CameraWorker(QThread):
    """Камера → BlazePose → счётчик → наложение → MP4."""

    frame_ready = pyqtSignal(object, dict)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        settings: dict[str, Any],
        bar_y: float | None,
        output_directory: Path,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.bar_y = bar_y
        self.output_directory = output_directory
        self._stop_event = threading.Event()

    def set_bar_y(self, value: float | None) -> None:
        """Обновляет координату калибровочной линии."""
        self.bar_y = value

    def stop(self) -> None:
        """Запрашивает завершение цикла видеозахвата."""
        self._stop_event.set()

    def run(self) -> None:
        capture = None
        processor = None
        recorder = None

        try:
            # DirectShow требуется ТЗ для работы под Windows.
            capture = cv2.VideoCapture(
                self.settings["camera_index"],
                cv2.CAP_DSHOW,
            )

            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(
                    self.settings["camera_index"]
                )

            capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.settings["frame_width"],
            )
            capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.settings["frame_height"],
            )

            if not capture.isOpened():
                raise RuntimeError(
                    f"Камера {self.settings['camera_index']} "
                    "не найдена или занята"
                )

            width = int(
                capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            ) or 640
            height = int(
                capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ) or 480
            fps = capture.get(cv2.CAP_PROP_FPS)

            processor = PoseProcessor(smoothing_window=5)
            counter = RepCounter(
                swing_angle_threshold=self.settings[
                    "swing_angle_threshold"
                ],
                swing_deviation_ratio=self.settings[
                    "swing_deviation_ratio"
                ],
            )

            try:
                filename = (
                    f"workout_"
                    f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
                )
                recorder = VideoRecorder(
                    self.output_directory / filename,
                    width,
                    height,
                    fps,
                )
                self.status_changed.emit(
                    f"Запись: {recorder.path.name}"
                )

            except Exception as exc:
                # Ошибка записи не останавливает тренировку.
                recorder = None
                self.status_changed.emit(
                    f"Видеозапись недоступна: {exc}"
                )

            started = time.monotonic()
            failed_reads = 0

            while not self._stop_event.is_set():
                ok, frame = capture.read()

                if not ok:
                    failed_reads += 1

                    if failed_reads >= 15:
                        raise RuntimeError(
                            "Видеопоток прерван: "
                            "камера не возвращает кадры"
                        )
                    continue

                failed_reads = 0
                elapsed = time.monotonic() - started
                landmarks = processor.process(frame)

                if landmarks is None:
                    counter.handle_pose_lost()
                    active_errors = (
                        "ПОЛЬЗОВАТЕЛЬ НЕ ОБНАРУЖЕН",
                    )
                else:
                    analysis = counter.update(
                        landmarks,
                        width,
                        height,
                        self.bar_y,
                        elapsed,
                    )
                    active_errors = analysis.active_errors
                    processor.draw_skeleton(
                        frame,
                        landmarks,
                    )

                if self.bar_y is not None:
                    line_y = max(
                        0,
                        min(
                            height - 1,
                            int(self.bar_y),
                        ),
                    )
                    cv2.line(
                        frame,
                        (0, line_y),
                        (width, line_y),
                        (0, 220, 255),
                        2,
                    )

                stats = counter.snapshot()
                stats["elapsed_seconds"] = elapsed

                minutes, seconds = divmod(
                    int(elapsed),
                    60,
                )
                counts = stats["error_counts"]

                lines = [
                    (
                        f"Повторения: {stats['repetitions']}    "
                        f"Время: {minutes:02d}:{seconds:02d}",
                        (255, 255, 255),
                    ),
                    (
                        f"Ошибки: подъём "
                        f"{counts['incomplete_ascent']} | "
                        f"раскачивание {counts['swing']} | "
                        f"руки {counts['incomplete_descent']}",
                        (255, 255, 255),
                    ),
                ]

                if self.bar_y is None:
                    lines.append(
                        (
                            "ВЫПОЛНИТЕ КАЛИБРОВКУ ПЕРЕКЛАДИНЫ",
                            (255, 210, 40),
                        )
                    )

                lines.extend(
                    (warning, (255, 80, 80))
                    for warning in active_errors
                )

                draw_overlay(frame, lines)

                if recorder is not None:
                    recorder.write(frame)

                self.frame_ready.emit(frame, stats)

        except Exception as exc:
            traceback.print_exc()
            self.error_occurred.emit(str(exc))

        finally:
            if recorder is not None:
                try:
                    recorder.close()
                except Exception as exc:
                    self.error_occurred.emit(
                        "Не удалось завершить "
                        f"видеофайл: {exc}"
                    )

            if capture is not None:
                capture.release()

            if processor is not None:
                processor.close()


class MainWindow(QMainWindow):
    """Главное окно с видеопотоком и статистикой."""

    def __init__(self) -> None:
        super().__init__()

        self.base_directory = (
            Path(__file__).resolve().parent
        )
        self.settings_manager = SettingsManager(
            self.base_directory / "settings.json"
        )
        self.settings = self.settings_manager.load()
        self.bar_y = self.settings.get("bar_y")

        self.worker: CameraWorker | None = None
        self.logger: StatsLogger | None = None
        self.last_stats: dict[str, Any] | None = None
        self.session_active = False

        self.setWindowTitle("Турник-Ассистент")
        self.resize(
            self.settings["window_width"],
            self.settings["window_height"],
        )
        self.move(
            self.settings["window_x"],
            self.settings["window_y"],
        )

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()

        if self.settings_manager.last_error:
            QTimer.singleShot(
                0,
                lambda: QMessageBox.warning(
                    self,
                    "Настройки",
                    self.settings_manager.last_error,
                ),
            )

    def _build_ui(self) -> None:
        """Создаёт видеообласть и панель статистики."""
        self.video_label = VideoLabel()
        self.video_label.video_clicked.connect(
            self.on_video_clicked
        )

        self.start_button = QPushButton("Старт")
        self.stop_button = QPushButton("Стоп")
        self.calibrate_button = QPushButton(
            "Калибровка"
        )
        self.help_button = QPushButton("Справка")

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(
            self.start_camera
        )
        self.stop_button.clicked.connect(
            self.stop_camera
        )
        self.calibrate_button.clicked.connect(
            self.toggle_calibration
        )
        self.help_button.clicked.connect(
            self.show_help
        )

        self.repetitions_label = QLabel(
            "Повторения: 0"
        )
        self.repetitions_label.setStyleSheet(
            "font-size: 26px; font-weight: 600;"
        )
        self.timer_label = QLabel("Время: 00:00")
        self.errors_label = QLabel(
            "Неполный подъём: 0\n"
            "Раскачивание: 0\n"
            "Неполное опускание: 0"
        )
        self.event_label = QLabel("Камера выключена")
        self.event_label.setWordWrap(True)

        buttons = QHBoxLayout()

        for button in (
            self.start_button,
            self.stop_button,
            self.calibrate_button,
            self.help_button,
        ):
            buttons.addWidget(button)

        panel = QVBoxLayout()
        panel.addWidget(self.repetitions_label)
        panel.addWidget(self.timer_label)

        separator = QFrame()
        separator.setFrameShape(
            QFrame.Shape.HLine
        )
        panel.addWidget(separator)
        panel.addWidget(self.errors_label)
        panel.addWidget(self.event_label)
        panel.addStretch(1)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setMinimumWidth(250)

        splitter = QSplitter()
        splitter.addWidget(self.video_label)
        splitter.addWidget(panel_widget)
        splitter.setStretchFactor(0, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(buttons)
        layout.addWidget(splitter, stretch=1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Готов к работе"
        )

    def _build_menu(self) -> None:
        """Добавляет команды отчёта, выхода и справки."""
        file_menu = self.menuBar().addMenu("Файл")

        save_action = QAction(
            "Сохранить отчёт",
            self,
        )
        save_action.setShortcut(
            QKeySequence(
                QKeySequence.StandardKey.Save
            )
        )
        save_action.triggered.connect(
            self.save_report_dialog
        )

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(save_action)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu(
            "Справка"
        )
        help_action = QAction(
            "Как пользоваться",
            self,
        )
        help_action.setShortcut(
            QKeySequence(
                QKeySequence.StandardKey.HelpContents
            )
        )
        help_action.triggered.connect(
            self.show_help
        )
        help_menu.addAction(help_action)

    def _build_shortcuts(self) -> None:
        """Регистрирует горячие клавиши."""
        self.shortcut_toggle = QShortcut(
            QKeySequence("Space"),
            self,
        )
        self.shortcut_toggle.activated.connect(
            self.toggle_camera
        )

        self.shortcut_calibrate = QShortcut(
            QKeySequence("C"),
            self,
        )
        self.shortcut_calibrate.activated.connect(
            self.toggle_calibration
        )

        self.shortcut_cancel = QShortcut(
            QKeySequence("Escape"),
            self,
        )
        self.shortcut_cancel.activated.connect(
            self.cancel_calibration
        )

    def toggle_camera(self) -> None:
        """Запускает или останавливает захват."""
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        """Создаёт новую сессию и рабочий поток."""
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        self.logger = StatsLogger()
        self.last_stats = None
        self.session_active = True

        self.worker = CameraWorker(
            self.settings,
            self.bar_y,
            self.base_directory / "recordings",
        )
        self.worker.frame_ready.connect(
            self.on_frame_ready
        )
        self.worker.error_occurred.connect(
            self.on_worker_error
        )
        self.worker.status_changed.connect(
            self.statusBar().showMessage
        )
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.event_label.setText(
            "Камера запущена"
        )

    def stop_camera(self) -> bool:
        """Останавливает поток и освобождает ресурсы."""
        if self.worker is None:
            return True

        self.stop_button.setEnabled(False)
        self.worker.stop()

        if not self.worker.wait(7000):
            # Нельзя удалять работающий QThread.
            self.stop_button.setEnabled(True)
            self.on_worker_error(
                "Поток камеры не завершился за 7 секунд"
            )
            return False

        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(
            "Видеозахват остановлен"
        )
        return True

    def on_frame_ready(
        self,
        frame: np.ndarray,
        stats: dict[str, Any],
    ) -> None:
        """Обновляет изображение и счётчики в GUI-потоке."""
        self.video_label.set_frame(frame)
        self.last_stats = stats

        self.repetitions_label.setText(
            f"Повторения: {stats['repetitions']}"
        )

        minutes, seconds = divmod(
            int(stats["elapsed_seconds"]),
            60,
        )
        self.timer_label.setText(
            f"Время: {minutes:02d}:{seconds:02d}"
        )

        counts = stats["error_counts"]
        self.errors_label.setText(
            f"Неполный подъём: "
            f"{counts['incomplete_ascent']}\n"
            f"Раскачивание: {counts['swing']}\n"
            f"Неполное опускание: "
            f"{counts['incomplete_descent']}"
        )

        if stats["last_event"]:
            self.event_label.setText(
                stats["last_event"]
            )

    def on_worker_error(self, message: str) -> None:
        """Показывает ошибку без потери текущей статистики."""
        self.statusBar().showMessage(message)
        self.event_label.setText(message)

    def toggle_calibration(self) -> None:
        """Включает режим калибровки следующим кликом."""
        if self.video_label.pixmap() is None:
            QMessageBox.information(
                self,
                "Калибровка",
                "Сначала запустите камеру.",
            )
            return

        enabled = (
            not self.video_label.calibration_enabled
        )
        self.video_label.calibration_enabled = enabled
        self.video_label.setCursor(
            Qt.CursorShape.CrossCursor
            if enabled
            else Qt.CursorShape.ArrowCursor
        )
        self.calibrate_button.setText(
            "Отмена калибровки"
            if enabled
            else "Калибровка"
        )

        if enabled:
            self.statusBar().showMessage(
                "Кликните по горизонтальной перекладине"
            )

    def cancel_calibration(self) -> None:
        """Отменяет калибровку без изменения значения."""
        if self.video_label.calibration_enabled:
            self.toggle_calibration()
            self.statusBar().showMessage(
                "Калибровка отменена"
            )

    def on_video_clicked(
        self,
        _x: int,
        y: int,
    ) -> None:
        """Сохраняет Y перекладины и передаёт его потоку."""
        self.bar_y = y
        self.settings["bar_y"] = y

        if self.worker is not None:
            self.worker.set_bar_y(y)

        self.settings_manager.save(self.settings)
        self.cancel_calibration()
        self.statusBar().showMessage(
            f"Перекладина откалибрована: Y = {y}"
        )

    def save_report_dialog(self) -> None:
        """Сохраняет отчёт в выбранный TXT-файл."""
        if not self.session_active:
            QMessageBox.information(
                self,
                "Отчёт",
                "Сначала выполните тренировку.",
            )
            return

        default_path = (
            self.base_directory / "report.txt"
        )
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчёт",
            str(default_path),
            "Текстовые файлы (*.txt)",
        )

        if filename:
            self._save_report(Path(filename))

    def _save_report(self, path: Path) -> bool:
        """Передаёт статистику StatsLogger."""
        if self.logger is None:
            return False

        stats = self.last_stats or {
            "repetitions": 0,
            "error_counts": {
                key: 0 for key in ERROR_TITLES
            },
            "error_log": [],
            "elapsed_seconds": 0.0,
        }

        try:
            saved_path = self.logger.save_report(
                path,
                stats["repetitions"],
                stats["error_counts"],
                stats["elapsed_seconds"],
                stats["error_log"],
            )
            self.statusBar().showMessage(
                f"Отчёт сохранён: {saved_path}"
            )
            return True

        except OSError as exc:
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить отчёт: {exc}",
            )
            return False

    def show_help(self) -> None:
        """Открывает встроенную справку."""
        QMessageBox.information(
            self,
            "Справка — Турник-Ассистент",
            "Приложение анализирует подтягивания "
            "с веб-камеры (вид сбоку).\n\n"
            "1. Нажмите «Старт» и расположитесь так, "
            "чтобы были видны нос, руки и таз.\n"
            "2. Нажмите «Калибровка» и кликните "
            "по перекладине.\n"
            "3. Полное повторение: выпрямленные руки "
            "внизу → нос выше линии → выпрямленные "
            "руки внизу.\n\n"
            "Ошибки: нос не пересёк линию; "
            "раскачивание; угол локтя внизу меньше 170°.\n"
            "Повторение с любой ошибкой не засчитывается.\n\n"
            "Горячие клавиши: Space — старт/стоп, "
            "C — калибровка, Esc — отмена, "
            "Ctrl+S — отчёт, F1 — справка.\n"
            "Видео сохраняется в recordings. "
            "Отчёт также создаётся при закрытии программы.",
        )

    def _persist_settings(self) -> None:
        """Сохраняет положение окна и калибровку."""
        geometry = self.geometry()

        self.settings.update(
            {
                "window_x": geometry.x(),
                "window_y": geometry.y(),
                "window_width": geometry.width(),
                "window_height": geometry.height(),
                "bar_y": self.bar_y,
            }
        )
        self.settings_manager.save(self.settings)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Сохраняет данные и освобождает ресурсы."""
        if not self.stop_camera():
            event.ignore()
            return

        self._persist_settings()

        if self.session_active:
            self._save_report(
                self.base_directory / "report.txt"
            )

        event.accept()