"""Распознавание BlazePose, сглаживание координат и отрисовка скелета."""

from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class PosePoint:
    """Нормализованная координата ключевой точки и её уверенность."""

    x: float
    y: float
    visibility: float


class PoseProcessor:
    """Инкапсулирует MediaPipe Pose и скользящее среднее по кадрам."""

    def __init__(self, smoothing_window: int = 5) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._history: dict[str, deque[tuple[float, float]]] = {
            landmark.name.lower(): deque(
                maxlen=max(3, smoothing_window)
            )
            for landmark in self._mp_pose.PoseLandmark
        }

    def process(
        self,
        frame_bgr: np.ndarray,
    ) -> dict[str, PosePoint] | None:
        """Возвращает сглаженные точки либо None, если человек не найден."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            self.clear_history()
            return None

        smoothed: dict[str, PosePoint] = {}

        for index, landmark in enumerate(
            result.pose_landmarks.landmark
        ):
            name = self._mp_pose.PoseLandmark(index).name.lower()
            history = self._history[name]
            history.append((landmark.x, landmark.y))

            smoothed[name] = PosePoint(
                x=sum(point[0] for point in history) / len(history),
                y=sum(point[1] for point in history) / len(history),
                visibility=landmark.visibility,
            )

        return smoothed

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        landmarks: dict[str, PosePoint],
        min_visibility: float = 0.5,
    ) -> None:
        """Рисует соединения BlazePose и ключевые точки поверх кадра."""
        height, width = frame_bgr.shape[:2]

        def pixel(name: str) -> tuple[int, int] | None:
            point = landmarks.get(name)

            if point is None or point.visibility < min_visibility:
                return None

            return int(point.x * width), int(point.y * height)

        for start, end in self._mp_pose.POSE_CONNECTIONS:
            start_name = self._mp_pose.PoseLandmark(
                start
            ).name.lower()
            end_name = self._mp_pose.PoseLandmark(end).name.lower()

            start_xy = pixel(start_name)
            end_xy = pixel(end_name)

            if start_xy is not None and end_xy is not None:
                cv2.line(
                    frame_bgr,
                    start_xy,
                    end_xy,
                    (80, 220, 80),
                    2,
                    cv2.LINE_AA,
                )

        for name in landmarks:
            xy = pixel(name)
            if xy is not None:
                cv2.circle(
                    frame_bgr,
                    xy,
                    4,
                    (0, 80, 255),
                    -1,
                    cv2.LINE_AA,
                )

    def clear_history(self) -> None:
        """Исключает влияние старой позы на новое обнаружение."""
        for history in self._history.values():
            history.clear()

    def close(self) -> None:
        """Освобождает внутренние ресурсы MediaPipe."""
        self._pose.close()