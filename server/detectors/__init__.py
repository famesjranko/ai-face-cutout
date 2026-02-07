from server.detectors.base import BaseDetector, DetectionResult
from server.detectors.face_bisenet import FaceBiSeNetDetector
from server.detectors.yolov8_seg import YOLOv8SegDetector
from server.enums import DetectionMode

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "registry",
    "create_detector",
]

# Maps DetectionMode value -> detector class.
# Adding a new detector only requires:
#   1. Implementing BaseDetector with detection_mode()
#   2. Adding a _SETTINGS_MAP entry below
#   3. Importing the class here
registry = {
    cls.detection_mode(): cls
    for cls in [FaceBiSeNetDetector, YOLOv8SegDetector]
}

# Maps DetectionMode value -> callable(settings) returning constructor kwargs.
# This keeps config-to-constructor mapping in one place so app.py stays generic.
_SETTINGS_MAP = {
    DetectionMode.FACE: lambda s: dict(
        yolo_weights=s.WEIGHTS_PATH,
        bisenet_weights=s.BISENET_WEIGHTS_PATH,
        device=s.DEVICE,
        img_size=s.IMG_SIZE,
    ),
    DetectionMode.OBJECT: lambda s: dict(
        model_variant=s.YOLOV8_SEG_MODEL,
        device=s.DEVICE,
    ),
}


def create_detector(mode, settings=None, **kwargs):
    """Instantiate a detector by mode.

    If *settings* is provided, constructor kwargs are derived from the
    settings object via ``_SETTINGS_MAP``.  Extra *kwargs* are merged in
    (and override settings-derived values).
    """
    cls = registry.get(mode)
    if cls is None:
        raise ValueError(f"Unknown detection mode: {mode}")
    if settings is not None and mode in _SETTINGS_MAP:
        base_kwargs = _SETTINGS_MAP[mode](settings)
        base_kwargs.update(kwargs)
        return cls(**base_kwargs)
    return cls(**kwargs)
