"""Геометрический анализ позы и автомат подсчёта подтягиваний."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


ERROR_LABELS = {
    "incomplete_ascent": "НЕПОЛНЫЙ ПОДЪЁМ!",
    "swing": "РАСКАЧИВАНИЕ!",
    "incomplete_descent": "ВЫПРЯМИ РУКИ!",
}


@dataclass(frozen=True)
class AnalysisResult:
    """Результат обработки одного кадра для наложения на видео."""

    elbow_angle: float | None
    active_errors: tuple[str, ...]
    state: str


class RepCounter:
    """Засчитывает только полный цикл «вниз–вверх–вниз» без ошибок."""

    DOWN_ANGLE = 170.0
    LEFT_BOTTOM_ANGLE = 155.0
    REVERSAL_DELTA = 15.0
    WARNING_SECONDS = 1.5

    def __init__(
        self,
        swing_angle_threshold: float = 5.0,
        swing_deviation_ratio: float = 0.15,
    ) -> None:
        self.swing_angle_threshold = swing_angle_threshold
        self.swing_deviation_ratio = swing_deviation_ratio

        self.repetitions = 0
        self.error_counts = {key: 0 for key in ERROR_LABELS}
        self.error_log: list[dict[str, Any]] = []

        self.state = "WAIT_BOTTOM"
        self.last_event = ""

        self._current_rep_errors: set[str] = set()
        self._error_latched: set[str] = set()
        self._warning_until = {
            key: -1.0 for key in ERROR_LABELS
        }

        self._torso_angles: deque[tuple[float, float]] = deque()
        self._left_bottom = False
        self._max_descent_angle = 0.0

    @staticmethod
    def calculate_angle(
        shoulder: tuple[float, float],
        elbow: tuple[float, float],
        wrist: tuple[float, float],
    ) -> float | None:
        """Вычисляет угол локтя по теореме косинусов."""
        a = math.dist(shoulder, elbow)
        b = math.dist(elbow, wrist)
        c = math.dist(shoulder, wrist)

        if a < 1e-6 or b < 1e-6:
            return None

        cosine = max(
            -1.0,
            min(
                1.0,
                (a * a + b * b - c * c) / (2 * a * b),
            ),
        )
        return math.degrees(math.acos(cosine))

    def update(
        self,
        landmarks: dict[str, Any],
        frame_width: int,
        frame_height: int,
        bar_y: float | None,
        elapsed: float,
    ) -> AnalysisResult:
        """Обновляет фазу движения и фиксирует ошибки."""
        seen_errors: set[str] = set()

        if self._is_swinging(
            landmarks,
            frame_width,
            frame_height,
            elapsed,
        ):
            self._register_error("swing", elapsed)
            seen_errors.add("swing")

        angle = self._visible_elbow_angle(
            landmarks,
            frame_width,
            frame_height,
        )

        if angle is not None and bar_y is not None:
            nose = landmarks.get("nose")

            if nose is not None and nose.visibility >= 0.35:
                above_bar = nose.y * frame_height <= bar_y
                self._update_state(
                    angle,
                    above_bar,
                    elapsed,
                    seen_errors,
                )

        # Раскачивание не регистрируется повторно, пока условие активно.
        self._error_latched.intersection_update(seen_errors)

        return AnalysisResult(
            elbow_angle=angle,
            active_errors=self.get_active_errors(elapsed),
            state=self.state,
        )

    def _update_state(
        self,
        angle: float,
        above_bar: bool,
        elapsed: float,
        seen_errors: set[str],
    ) -> None:
        """Выполняет переходы между фазами упражнения."""
        if self.state == "WAIT_BOTTOM":
            if angle >= self.DOWN_ANGLE:
                self._begin_attempt()
            return

        if self.state == "ASCENDING":
            if angle < self.LEFT_BOTTOM_ANGLE:
                self._left_bottom = True

            if self._left_bottom and above_bar:
                self.state = "DESCENDING"
                self._max_descent_angle = angle

            elif self._left_bottom and angle >= self.DOWN_ANGLE:
                # Возврат вниз без пересечения перекладины.
                self._register_error(
                    "incomplete_ascent",
                    elapsed,
                )
                seen_errors.add("incomplete_ascent")
                self.last_event = (
                    "Повторение не засчитано: неполный подъём"
                )
                self._begin_attempt()

            return

        if self.state == "DESCENDING":
            self._max_descent_angle = max(
                self._max_descent_angle,
                angle,
            )

            if angle >= self.DOWN_ANGLE:
                if self._current_rep_errors:
                    self.last_event = (
                        "Повторение не засчитано: были ошибки"
                    )
                else:
                    self.repetitions += 1
                    self.last_event = "Повторение засчитано"

                self._begin_attempt()

            elif (
                self._max_descent_angle < self.DOWN_ANGLE
                and angle
                < self._max_descent_angle - self.REVERSAL_DELTA
            ):
                # Новый подъём начался до полного разгибания рук.
                self._register_error(
                    "incomplete_descent",
                    elapsed,
                )
                seen_errors.add("incomplete_descent")
                self.last_event = (
                    "Повторение не засчитано: неполное опускание"
                )
                self._reset_attempt("WAIT_BOTTOM")

    def _begin_attempt(self) -> None:
        """Начинает попытку из выпрямленной нижней точки."""
        self.state = "ASCENDING"
        self._current_rep_errors.clear()
        self._left_bottom = False
        self._max_descent_angle = 0.0

    def _reset_attempt(self, state: str) -> None:
        """Сбрасывает незавершённую попытку."""
        self.state = state
        self._current_rep_errors.clear()
        self._left_bottom = False
        self._max_descent_angle = 0.0

    def _register_error(
        self,
        error_type: str,
        elapsed: float,
    ) -> None:
        """Добавляет ошибку в попытку и однократно — в журнал."""
        self._current_rep_errors.add(error_type)
        self._warning_until[error_type] = (
            elapsed + self.WARNING_SECONDS
        )

        if error_type not in self._error_latched:
            self.error_counts[error_type] += 1
            self.error_log.append(
                {
                    "time": round(elapsed, 2),
                    "type": error_type,
                }
            )
            self._error_latched.add(error_type)

    def _visible_elbow_angle(
        self,
        landmarks: dict[str, Any],
        width: int,
        height: int,
    ) -> float | None:
        """Выбирает наиболее видимую руку для анализа вида сбоку."""
        candidates: list[tuple[float, float]] = []

        for side in ("left", "right"):
            names = [
                f"{side}_{part}"
                for part in ("shoulder", "elbow", "wrist")
            ]
            points = [landmarks.get(name) for name in names]

            if any(point is None for point in points):
                continue

            visibility = (
                sum(point.visibility for point in points) / 3
            )

            if visibility < 0.35:
                continue

            shoulder, elbow, wrist = points
            angle = self.calculate_angle(
                (
                    shoulder.x * width,
                    shoulder.y * height,
                ),
                (
                    elbow.x * width,
                    elbow.y * height,
                ),
                (
                    wrist.x * width,
                    wrist.y * height,
                ),
            )

            if angle is not None:
                candidates.append((visibility, angle))

        return max(candidates)[1] if candidates else None

    def _is_swinging(
        self,
        landmarks: dict[str, Any],
        width: int,
        height: int,
        elapsed: float,
    ) -> bool:
        """Проверяет отклонение таза и амплитуду наклона туловища."""
        names = (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        )
        points = [landmarks.get(name) for name in names]

        if any(
            point is None or point.visibility < 0.35
            for point in points
        ):
            return False

        (
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        ) = points

        shoulder_x = (
            (left_shoulder.x + right_shoulder.x) * width / 2
        )
        shoulder_y = (
            (left_shoulder.y + right_shoulder.y) * height / 2
        )
        hip_x = (left_hip.x + right_hip.x) * width / 2
        hip_y = (left_hip.y + right_hip.y) * height / 2

        shoulder_width = math.dist(
            (
                left_shoulder.x * width,
                left_shoulder.y * height,
            ),
            (
                right_shoulder.x * width,
                right_shoulder.y * height,
            ),
        )

        deviation = abs(hip_x - shoulder_x)
        horizontal_swing = (
            shoulder_width >= 10
            and deviation
            > shoulder_width * self.swing_deviation_ratio
        )

        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        torso_angle = math.degrees(
            math.atan2(abs(dx), max(abs(dy), 1e-6))
        )

        self._torso_angles.append((elapsed, torso_angle))

        while (
            self._torso_angles
            and elapsed - self._torso_angles[0][0] > 1.0
        ):
            self._torso_angles.popleft()

        angles = [
            value for _, value in self._torso_angles
        ]

        angular_swing = (
            len(angles) >= 3
            and max(angles) - min(angles)
            > self.swing_angle_threshold
        )

        return horizontal_swing or angular_swing

    def handle_pose_lost(self) -> None:
        """Не соединяет одну попытку из кадров до и после потери позы."""
        self._reset_attempt("WAIT_BOTTOM")
        self._torso_angles.clear()
        self.last_event = "Пользователь не обнаружен"

    def get_active_errors(
        self,
        elapsed: float,
    ) -> tuple[str, ...]:
        """Возвращает предупреждения, которые ещё нужно показывать."""
        return tuple(
            ERROR_LABELS[key]
            for key in ERROR_LABELS
            if self._warning_until[key] >= elapsed
        )

    def snapshot(self) -> dict[str, Any]:
        """Возвращает независимую копию статистики."""
        return {
            "repetitions": self.repetitions,
            "error_counts": self.error_counts.copy(),
            "error_log": [
                event.copy() for event in self.error_log
            ],
            "state": self.state,
            "last_event": self.last_event,
        }