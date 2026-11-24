from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameData:
    image: np.ndarray
    index: int
    timestamp_ms: float


class VideoReader:
    def __init__(self, video_path: str | Path, frame_step: int = 1, max_frames: int | None = None):
        self.path = Path(video_path)
        self.frame_step = max(1, frame_step)
        self.max_frames = max_frames
        self.cap = cv2.VideoCapture(str(self.path))

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps > 1.0 and not np.isnan(fps) else 30.0

    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        read_index = 0
        yielded = 0

        while True:
            if self.max_frames is not None and yielded >= self.max_frames:
                break

            ok, frame = self.cap.read()
            if not ok:
                break

            if read_index % self.frame_step == 0:
                ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                yield FrameData(image=frame, index=yielded, timestamp_ms=ts)
                yielded += 1

            read_index += 1

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()
