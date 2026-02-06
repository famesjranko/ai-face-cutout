"""Face segmentation detector: YOLOv5-Face bbox + BiSeNet pixel-level mask."""

import logging
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from server.detectors.base import BaseDetector, DetectionResult
from server.detectors.bisenet_model import BiSeNet

logger = logging.getLogger(__name__)

_BISENET_INPUT_SIZE = 512

# BiSeNet face-parsing class indices that belong to the face region.
# 0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
# 6=eye_g (glasses), 7=l_ear, 8=r_ear, 9=earring, 10=nose,
# 11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=necklace, 16=cloth, 17=hair, 18=hat
_FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13}

# Overlay colours per class for the annotated preview (BGR).
_CLASS_COLORS = {
    1: (0, 153, 255),    # skin — warm orange
    2: (50, 100, 200),   # l_brow
    3: (50, 100, 200),   # r_brow
    4: (200, 200, 0),    # l_eye — cyan
    5: (200, 200, 0),    # r_eye
    6: (200, 200, 200),  # glasses
    7: (0, 200, 200),    # l_ear
    8: (0, 200, 200),    # r_ear
    10: (0, 180, 100),   # nose
    11: (0, 0, 220),     # mouth
    12: (80, 0, 200),    # upper lip
    13: (80, 0, 200),    # lower lip
}

_BISENET_DOWNLOAD_URL = (
    "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth"
)

# ImageNet normalisation used by the pretrained BiSeNet.
_NORMALISE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class FaceBiSeNetDetector(BaseDetector):
    """Detects faces with YOLOv5-Face, then segments each with BiSeNet."""

    def __init__(
        self,
        yolo_weights: str = "weights/yolov5n-0.5.pt",
        bisenet_weights: str = "weights/bisenet_face_parsing.pth",
        device: str = "cpu",
        img_size: int = 320,
    ):
        self._yolo_weights = yolo_weights
        self._bisenet_weights = bisenet_weights
        self._device_str = device
        self._img_size = img_size
        self._yolo_model = None
        self._bisenet = None

    # -- BaseDetector interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "face"

    @property
    def is_loaded(self) -> bool:
        return self._yolo_model is not None and self._bisenet is not None

    def load(self) -> None:
        from server.detection import load_model  # avoid circular at import time

        logger.info("Loading YOLOv5-Face weights from %s ...", self._yolo_weights)
        self._yolo_model = load_model(self._yolo_weights, self._device_str)

        self._ensure_bisenet_weights()
        logger.info("Loading BiSeNet weights from %s ...", self._bisenet_weights)
        dev = torch.device(self._device_str)
        self._bisenet = BiSeNet(n_classes=19)
        self._bisenet.load_state_dict(
            torch.load(self._bisenet_weights, map_location=dev, weights_only=True),
            strict=False,
        )
        self._bisenet.to(dev)
        self._bisenet.eval()
        logger.info("FaceBiSeNetDetector ready.")

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        from server.detection import detect_frame

        annotated, det = detect_frame(
            self._yolo_model, frame_bgr, self._img_size, self._device_str
        )
        h, w = frame_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        if len(det) == 0:
            return DetectionResult(
                annotated_frame=annotated,
                mask=full_mask,
                count=0,
                label="0 faces",
            )

        for *xyxy, _conf, _cls in det:
            x1 = int(xyxy[0].item())
            y1 = int(xyxy[1].item())
            x2 = int(xyxy[2].item())
            y2 = int(xyxy[3].item())
            # Pad crop by 30% of bbox size for better segmentation context.
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
            cx1 = max(x1 - pad_x, 0)
            cy1 = max(y1 - pad_y, 0)
            cx2 = min(x2 + pad_x, w)
            cy2 = min(y2 + pad_y, h)

            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            seg_map = self._run_bisenet(crop)  # HxW int64 class map
            face_region = np.isin(seg_map, list(_FACE_CLASSES)).astype(np.uint8) * 255

            # Map back to full frame coordinates.
            full_mask[cy1:cy2, cx1:cx2] = np.maximum(
                full_mask[cy1:cy2, cx1:cx2], face_region
            )

        # Draw coloured segmentation overlay on annotated frame.
        annotated = self._draw_overlay(annotated, full_mask)

        count = len(det)
        label = f"{count} face{'s' if count != 1 else ''}"
        return DetectionResult(
            annotated_frame=annotated,
            mask=full_mask,
            count=count,
            label=label,
        )

    # -- Internal helpers ------------------------------------------------------

    def _ensure_bisenet_weights(self):
        if os.path.isfile(self._bisenet_weights):
            return
        os.makedirs(os.path.dirname(self._bisenet_weights) or ".", exist_ok=True)
        logger.info("Downloading BiSeNet weights to %s ...", self._bisenet_weights)
        torch.hub.download_url_to_file(_BISENET_DOWNLOAD_URL, self._bisenet_weights)

    @torch.no_grad()
    def _run_bisenet(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Run BiSeNet on a BGR crop, return HxW class-index map."""
        dev = torch.device(self._device_str)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(crop_rgb, (_BISENET_INPUT_SIZE, _BISENET_INPUT_SIZE))

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        tensor = _NORMALISE(tensor).unsqueeze(0).to(dev)

        out = self._bisenet(tensor)[0]  # take main output
        parsed = out.squeeze(0).argmax(0).cpu().numpy()  # (512, 512)

        # Resize back to crop dimensions.
        ch, cw = crop_bgr.shape[:2]
        parsed = cv2.resize(
            parsed.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST
        )
        return parsed

    @staticmethod
    def _draw_overlay(annotated: np.ndarray, full_mask: np.ndarray) -> np.ndarray:
        """Blend a semi-transparent coloured overlay where the mask is active."""
        overlay = annotated.copy()
        colour = np.array([180, 120, 255], dtype=np.uint8)  # purple-ish
        region = full_mask > 0
        overlay[region] = (
            overlay[region].astype(np.float32) * 0.6
            + colour.astype(np.float32) * 0.4
        ).astype(np.uint8)
        return overlay
