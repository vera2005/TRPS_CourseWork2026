"""Загрузка, проверка и атомарное сохранение настроек приложения."""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "bar_y": None,
    "window_x": 100,
    "window_y": 100,
    "window_width": 1100,
    "window_height": 720,
    "swing_angle_threshold": 5.0,
    "swing_deviation_ratio": 0.15,
}


class SettingsManager:
    """Читает settings.json и безопасно восстанавливается после повреждения."""

    def __init__(self, path: str | Path = "settings.json") -> None:
        self.path = Path(path)
        self.last_error = ""

    def load(self) -> dict[str, Any]:
        """Возвращает настройки, объединённые со значениями по умолчанию."""
        settings = deepcopy(DEFAULT_SETTINGS)
        self.last_error = ""

        try:
            with self.path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if not isinstance(loaded, dict):
                raise ValueError("Корневой элемент JSON должен быть объектом")

            settings.update(loaded)

        except FileNotFoundError:
            return settings

        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.last_error = (
                f"Файл настроек повреждён или недоступен: {exc}. "
                "Используются значения по умолчанию."
            )
            return settings

        return self._validate(settings)

    def save(self, settings: dict[str, Any]) -> bool:
        """Атомарно записывает настройки, чтобы сбой не обнулил JSON."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")

            with temporary.open("w", encoding="utf-8") as file:
                json.dump(settings, file, ensure_ascii=False, indent=2)

            os.replace(temporary, self.path)
            self.last_error = ""
            return True

        except OSError as exc:
            self.last_error = f"Не удалось сохранить настройки: {exc}"
            return False

    @staticmethod
    def _validate(settings: dict[str, Any]) -> dict[str, Any]:
        """Отбрасывает неверные типы и ограничивает параметры камеры."""
        validated = deepcopy(DEFAULT_SETTINGS)

        for key in ("camera_index", "frame_width", "frame_height"):
            value = settings.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                validated[key] = value

        validated["camera_index"] = max(
            0, min(validated["camera_index"], 32)
        )
        validated["frame_width"] = max(
            160, min(validated["frame_width"], 7680)
        )
        validated["frame_height"] = max(
            120, min(validated["frame_height"], 4320)
        )

        for key in (
            "window_x",
            "window_y",
            "window_width",
            "window_height",
        ):
            value = settings.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                validated[key] = value

        validated["window_width"] = max(640, validated["window_width"])
        validated["window_height"] = max(480, validated["window_height"])

        bar_y = settings.get("bar_y")
        if bar_y is None or (
            isinstance(bar_y, (int, float))
            and not isinstance(bar_y, bool)
            and bar_y >= 0
        ):
            validated["bar_y"] = bar_y

        for key, lower, upper in (
            ("swing_angle_threshold", 1.0, 45.0),
            ("swing_deviation_ratio", 0.01, 1.0),
        ):
            value = settings.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                validated[key] = max(lower, min(float(value), upper))

        return validated