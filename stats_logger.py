"""Формирование текстового отчёта и журнала ошибок тренировки."""

from datetime import datetime
from pathlib import Path
from typing import Iterable


ERROR_TITLES = {
    "incomplete_ascent": "Неполный подъём",
    "swing": "Раскачивание",
    "incomplete_descent": "Неполное опускание",
}


class StatsLogger:
    """Сохраняет итоговую статистику сессии в TXT с кодировкой UTF-8."""

    def __init__(self) -> None:
        self.started_at = datetime.now()

    def build_report(
        self,
        repetitions: int,
        error_counts: dict[str, int],
        duration_seconds: float,
        error_log: Iterable[dict],
    ) -> str:
        """Формирует отчёт, включая время каждой обнаруженной ошибки."""
        duration = max(0, int(duration_seconds))
        minutes, seconds = divmod(duration, 60)

        lines = [
            "ТУРНИК-АССИСТЕНТ — ОТЧЁТ О ТРЕНИРОВКЕ",
            f"Дата и время начала: {self.started_at:%d.%m.%Y %H:%M:%S}",
            f"Длительность: {minutes:02d}:{seconds:02d}",
            f"Засчитано повторений: {repetitions}",
            "",
            "Ошибки по типам:",
        ]

        for key, title in ERROR_TITLES.items():
            lines.append(f"- {title}: {error_counts.get(key, 0)}")

        lines.extend(("", "Журнал ошибок:"))
        events = list(error_log)

        if not events:
            lines.append("Ошибок не зафиксировано.")
        else:
            for event in events:
                elapsed = float(event.get("time", 0.0))
                title = ERROR_TITLES.get(
                    event.get("type", ""),
                    "Неизвестная ошибка",
                )
                lines.append(f"- {elapsed:8.2f} с — {title}")

        return "\n".join(lines) + "\n"

    def save_report(
        self,
        path: str | Path,
        repetitions: int,
        error_counts: dict[str, int],
        duration_seconds: float,
        error_log: Iterable[dict],
    ) -> Path:
        """Записывает отчёт и возвращает путь к созданному файлу."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.build_report(
            repetitions,
            error_counts,
            duration_seconds,
            error_log,
        )
        output_path.write_text(report, encoding="utf-8")
        return output_path