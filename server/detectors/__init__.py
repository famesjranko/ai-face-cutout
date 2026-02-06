from server.detectors.base import BaseDetector, DetectionResult
from server.detectors.face_bisenet import FaceBiSeNetDetector
from server.detectors.yolov8_seg import YOLOv8SegDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "FaceBiSeNetDetector",
    "YOLOv8SegDetector",
]
