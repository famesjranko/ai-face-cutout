import sys
import os
import copy

import cv2
import numpy as np
import torch

# Ensure project root is on the path so YOLOv5 local imports work
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords


def load_model(weights: str, device: str) -> torch.nn.Module:
    dev = torch.device(device)
    model = attempt_load(weights, map_location=dev)
    return model


def _scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    coords[:, 4].clamp_(0, img0_shape[1])
    coords[:, 5].clamp_(0, img0_shape[0])
    coords[:, 6].clamp_(0, img0_shape[1])
    coords[:, 7].clamp_(0, img0_shape[0])
    coords[:, 8].clamp_(0, img0_shape[1])
    coords[:, 9].clamp_(0, img0_shape[0])
    return coords


def _draw_detections(img, xyxy, conf, landmarks):
    h, w, _ = img.shape
    tl = 1
    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
    img = img.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i in range(5):
        px = int(landmarks[2 * i])
        py = int(landmarks[2 * i + 1])
        cv2.circle(img, (px, py), tl + 1, colors[i], -1)

    tf = max(tl - 1, 1)
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect_frame(model, frame_bgr: np.ndarray, img_size: int, device: str):
    """Run YOLOv5-face detection on a single BGR frame.

    Returns (annotated_bgr, detections_tensor) where detections_tensor has
    coordinates scaled to the original frame_bgr shape.
    """
    dev = torch.device(device)

    # Prepare image — same pipeline as original detect()
    orgimg = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img0 = copy.deepcopy(orgimg)
    h0, w0 = orgimg.shape[:2]
    r = img_size / max(h0, w0)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()

    img_tensor = torch.from_numpy(img).to(dev).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]

    pred = non_max_suppression_face(pred, conf_thres=0.6, iou_thres=0.5)

    annotated = frame_bgr.copy()
    det = pred[0]

    if len(det):
        det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame_bgr.shape).round()
        det[:, 5:15] = _scale_coords_landmarks(img_tensor.shape[2:], det[:, 5:15], frame_bgr.shape).round()

        for j in range(det.size()[0]):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            landmarks = det[j, 5:15].view(-1).tolist()
            annotated = _draw_detections(annotated, xyxy, conf, landmarks)

    return annotated, det
