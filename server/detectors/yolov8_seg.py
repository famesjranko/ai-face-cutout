"""Person / object instance segmentation using YOLOv8-seg."""

import logging
from typing import List

import cv2
import numpy as np

from server.detectors.base import BaseDetector, DetectionResult

logger = logging.getLogger(__name__)

# COCO class names for the subset we expose in the UI.
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
}


class YOLOv8SegDetector(BaseDetector):
    """Instance segmentation with YOLOv8-seg (ultralytics)."""

    def __init__(
        self,
        model_variant: str = "yolov8n-seg.pt",
        device: str = "cpu",
        target_classes: List[int] | None = None,
    ):
        self._model_variant = model_variant
        self._device_str = device
        self._target_classes = target_classes if target_classes is not None else [0]
        self._model = None

    # -- BaseDetector interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "object"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        from ultralytics import YOLO

        logger.info("Loading YOLOv8-seg model: %s ...", self._model_variant)
        self._model = YOLO(self._model_variant)
        logger.info("YOLOv8SegDetector ready.")

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        results = self._model(frame_bgr, verbose=False)
        result = results[0]

        annotated = frame_bgr.copy()
        count = 0

        if result.masks is not None and result.boxes is not None:
            masks_data = result.masks.data.cpu().numpy()   # (N, mh, mw)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)

            for i, cls_id in enumerate(classes):
                if cls_id not in self._target_classes:
                    continue
                count += 1

                # Resize individual mask to frame dimensions.
                seg = cv2.resize(
                    masks_data[i], (w, h), interpolation=cv2.INTER_LINEAR
                )
                binary = (seg > 0.5).astype(np.uint8) * 255
                full_mask = np.maximum(full_mask, binary)

                # Draw contour + label on annotated frame.
                contours, _ = cv2.findContours(
                    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

                x1, y1, x2, y2 = boxes[i]
                cls_name = COCO_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                label_text = f"{cls_name} {confs[i]:.2f}"
                cv2.putText(
                    annotated, label_text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )

        # Build friendly label.
        if self._target_classes == [0]:
            noun = "person" if count == 1 else "people"
        else:
            noun = "object" if count == 1 else "objects"
        label = f"{count} {noun}"

        return DetectionResult(
            annotated_frame=annotated,
            mask=full_mask,
            count=count,
            label=label,
        )

    # -- Public helpers --------------------------------------------------------

    def set_target_classes(self, class_ids: List[int]) -> None:
        """Update target classes at runtime (when the user changes the dropdown)."""
        self._target_classes = class_ids
