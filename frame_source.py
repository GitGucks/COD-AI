from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Type

import numpy as np


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class CaptureDeviceFrameSource(FrameSource):
    def __init__(
        self,
        *,
        device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        backend: str = "dshow",
    ) -> None:
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.backend = backend.strip().lower()
        self._cv2 = None
        self._cap = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("opencv-python is required: pip install opencv-python") from exc

        backend_flag = 0
        if self.backend in {"dshow", "directshow"} and hasattr(cv2, "CAP_DSHOW"):
            backend_flag = cv2.CAP_DSHOW
        elif self.backend in {"msmf"} and hasattr(cv2, "CAP_MSMF"):
            backend_flag = cv2.CAP_MSMF

        cap = (
            cv2.VideoCapture(self.device_index, backend_flag)
            if backend_flag
            else cv2.VideoCapture(self.device_index)
        )
        if not cap or not cap.isOpened():
            raise RuntimeError(
                f"Failed to open capture device index={self.device_index} backend={self.backend!r}.\n"
                "Make sure the Elgato is plugged in and no other app (e.g. 4K Capture Utility) is using it."
            )

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        self._cv2 = cv2
        self._cap = cap
        self._started = True

    def read(self) -> np.ndarray | None:
        if not self._started or self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return np.asarray(frame) if not isinstance(frame, np.ndarray) else frame

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._cap is not None:
                self._cap.release()
        finally:
            self._cap = None
            self._cv2 = None
            self._started = False
