import cv2
import numpy as np
from PIL import Image

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


def create_mask_preview(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a BGR visualization of the mask overlay (for browser display).

    Background (non-detected) is whited out; detected region shows through.
    mask: HxW uint8, 255 = detected region, 0 = background.
    """
    if mask is None or mask.max() == 0:
        return frame_bgr.copy()

    # White out the background: invert the mask to a 3-channel white overlay,
    # then add it to the frame so non-detected areas become white.
    inv = cv2.bitwise_not(mask)
    inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    preview = cv2.addWeighted(frame_bgr, 1.0, inv_bgr, 1.0, 0)
    return preview


def create_inpaint_inputs(frame_bgr: np.ndarray, mask: np.ndarray):
    """Create (PIL Image, PIL mask) ready for SD inpainting pipeline.

    mask: HxW uint8, 255 = detected region, 0 = background.

    SD inpainting convention: white = inpaint, black = keep.
    We invert the detector mask so the *background* (non-detected) is inpainted.

    Returns None if mask is empty.
    """
    if mask is None or mask.max() == 0:
        return None

    # Invert: detected region → black (keep), background → white (inpaint).
    inverted = cv2.bitwise_not(mask)

    # Mild dilation to feather edges around the kept region.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sd_mask = cv2.dilate(inverted, kernel, iterations=1)

    # Resize mask to SD input size.
    sd_mask = _resize_and_pad(sd_mask, pad_value=255)

    # Prepare source image (same resize + pad).
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = _resize_and_pad(frame_rgb, pad_value=255)

    image_pil = Image.fromarray(frame_rgb)
    mask_pil = Image.fromarray(sd_mask, mode="L")

    return image_pil, mask_pil
