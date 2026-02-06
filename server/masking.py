import cv2
import numpy as np
import skimage.morphology
from PIL import Image, ImageDraw

# Percentage to shift bounding boxes downward (matches original detect_face.py)
_PERCENTAGE_FACTOR = 2.5
_SD_SIZE = 512


def _resize_and_pad(img: np.ndarray, pad_value: int = 255) -> np.ndarray:
    """Resize preserving aspect ratio, pad to _SD_SIZE x _SD_SIZE.

    Padding is added equally on both sides of the shorter dimension.
    Default pad_value=255 (white) so padded areas become inpaint regions.
    """
    h, w = img.shape[:2]
    scale = min(_SD_SIZE / w, _SD_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    if len(img.shape) == 3:
        padded = np.full((_SD_SIZE, _SD_SIZE, img.shape[2]), pad_value, dtype=np.uint8)
    else:
        padded = np.full((_SD_SIZE, _SD_SIZE), pad_value, dtype=np.uint8)

    y_off = (_SD_SIZE - new_h) // 2
    x_off = (_SD_SIZE - new_w) // 2
    padded[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return padded


def _build_raw_mask(frame_bgr: np.ndarray, detections) -> np.ndarray:
    """Build a binary mask from detection bounding boxes.

    White = detected face region, Black = background.
    """
    mask = np.zeros_like(frame_bgr, dtype=np.uint8)
    img_height = mask.shape[0]

    for *xyxy, conf, cls in detections:
        x = int(xyxy[0].item())
        y = int(xyxy[1].item() + (img_height * (_PERCENTAGE_FACTOR / 100)))
        w = int(xyxy[2].item()) - int(xyxy[0].item())
        h = int(xyxy[3].item()) - int(xyxy[1].item())
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

    return mask


def create_mask_preview(frame_bgr: np.ndarray, detections) -> np.ndarray:
    """Return a BGR visualization of the mask overlay (for browser display).

    If no detections, returns a copy of the frame.
    """
    if detections is None or len(detections) == 0:
        return frame_bgr.copy()

    mask = _build_raw_mask(frame_bgr, detections)
    inverted = cv2.bitwise_not(mask)
    preview = cv2.addWeighted(frame_bgr, 1.0, inverted, 1.0, 0)
    return preview


def create_inpaint_inputs(frame_bgr: np.ndarray, detections):
    """Create (PIL Image, PIL mask) ready for SD inpainting pipeline.

    The mask is white where the model should inpaint (background) and black
    where the face should be preserved — matching SD inpainting convention.

    Returns None if no detections.
    """
    if detections is None or len(detections) == 0:
        return None

    mask = _build_raw_mask(frame_bgr, detections)

    # Invert so background is white (to be inpainted)
    inverted = cv2.bitwise_not(mask)
    combined = cv2.addWeighted(frame_bgr, 1.0, inverted, 1.0, 0)

    # Resize preserving aspect ratio, pad with white (inpaint region)
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    combined_rgb = _resize_and_pad(combined_rgb, pad_value=255)

    pil_mask_img = Image.fromarray(combined_rgb)

    # --- Morphological cleanup for clean alpha mask (from original) ---
    orig = pil_mask_img.copy()
    ImageDraw.floodfill(pil_mask_img, xy=(0, 0), value=(255, 0, 255), thresh=50)

    n = np.array(pil_mask_img)
    bg_mask = (n[:, :, 0:3] == [255, 0, 255]).all(2)

    strel = skimage.morphology.disk(13)
    cleaned = skimage.morphology.binary_closing(bg_mask, footprint=strel)
    cleaned = skimage.morphology.binary_dilation(cleaned, footprint=strel)

    # SD expects: white = inpaint, black = keep
    mask_l = (cleaned * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_l, mode="L")

    # Prepare the source image (same resize + pad as mask)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = _resize_and_pad(frame_rgb, pad_value=255)
    image_pil = Image.fromarray(frame_rgb)

    return image_pil, mask_pil
