from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """Unified result from any detector.

    annotated_frame: BGR ndarray with visual overlay drawn on it.
    mask: HxW uint8 ndarray — 255 where detected, 0 for background.
    count: number of detected instances.
    label: human-readable summary, e.g. "2 faces".
    """
    annotated_frame: np.ndarray
    mask: np.ndarray
    count: int
    label: str


class BaseDetector(ABC):
    """Common interface for all detection/segmentation backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this detector."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is ready for inference."""
