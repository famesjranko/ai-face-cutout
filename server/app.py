import asyncio
import base64
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.detectors import BaseDetector, create_detector
from server.enums import DetectionMode, ModelStatus
from server.masking import create_mask_preview
from server.inpainting import InpaintingEngine
from server.inpaint_orchestrator import InpaintOrchestrator
from server.schemas import DetectionResponse, ErrorResponse

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    detectors: Dict[str, BaseDetector] = field(default_factory=dict)
    detector_lock: threading.Lock = field(default_factory=threading.Lock)
    inpaint_engine: Optional[InpaintingEngine] = None
    latest_frame: Optional[np.ndarray] = None
    latest_mask: Optional[np.ndarray] = None
    captured_frame: Optional[np.ndarray] = None
    captured_mask: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    inpaint_status: str = ModelStatus.LOADING
    inpaint_status_detail: str = "Starting up..."


state = AppState()


def _load_inpaint_model():
    """Load SD inpainting model in background thread."""
    def on_status(status, detail):
        with state.lock:
            state.inpaint_status = status
            state.inpaint_status_detail = detail

    try:
        engine = InpaintingEngine(
            model_id=settings.INPAINT_MODEL,
            device=settings.DEVICE,
            num_steps=settings.INPAINT_STEPS,
            guidance_scale=settings.GUIDANCE_SCALE,
        )
        with state.lock:
            state.inpaint_engine = engine
        state.inpaint_engine.load(status_callback=on_status)
        with state.lock:
            state.inpaint_status = ModelStatus.READY
            state.inpaint_status_detail = "Model ready"
    except Exception:
        with state.lock:
            state.inpaint_status = ModelStatus.ERROR
            state.inpaint_status_detail = "Failed to load model"
        logger.exception("Failed to load inpainting model")


def _prewarm_face_detector():
    """Pre-load the face detector in a background thread."""
    try:
        get_detector_sync(DetectionMode.FACE)
    except Exception:
        logger.exception("Failed to pre-warm face detector")


def get_detector_sync(mode: str, target_classes=None) -> BaseDetector:
    """Get or create a detector by mode name (thread-safe, lazy-loading)."""
    with state.detector_lock:
        if mode not in state.detectors:
            extra = {}
            if target_classes is not None:
                extra["target_classes"] = target_classes
            det = create_detector(mode, settings=settings, **extra)
            det.load()
            state.detectors[mode] = det
        else:
            # Update target classes if the detector supports it.
            det = state.detectors[mode]
            if target_classes is not None and hasattr(det, "set_target_classes"):
                det.set_target_classes(target_classes)
        return state.detectors[mode]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm face detector in background (replaces eager YOLOv5 load).
    logger.info("Pre-warming face detector in background...")
    threading.Thread(target=_prewarm_face_detector, daemon=True).start()

    # Load SD inpainting model in background so server starts immediately.
    logger.info("Loading inpainting model in background (may take minutes on first run)...")
    threading.Thread(target=_load_inpaint_model, daemon=True).start()

    yield  # App is running

    # Shutdown
    state.detectors.clear()
    state.inpaint_engine = None


app = FastAPI(lifespan=lifespan)


def _encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_detect_params(query_string: str):
    """Parse mode and classes from WebSocket query string."""
    from urllib.parse import parse_qs
    params = parse_qs(query_string)
    mode = params.get("mode", [settings.DEFAULT_DETECTION_MODE])[0]
    classes_str = params.get("classes", ["0"])[0]
    try:
        target_classes = [int(c) for c in classes_str.split(",") if c.strip()]
    except ValueError:
        target_classes = [0]
    return mode, target_classes


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()

    # Parse mode from query params.
    query = ws.scope.get("query_string", b"").decode("utf-8")
    mode, target_classes = _parse_detect_params(query)

    # Lazy-load the requested detector (may block on first use).
    loop = asyncio.get_event_loop()
    try:
        detector = await loop.run_in_executor(
            None, get_detector_sync, mode, target_classes
        )
    except Exception as exc:
        await ws.send_text(ErrorResponse(error=f"Failed to load detector: {exc}").model_dump_json())
        await ws.close()
        return

    try:
        while True:
            try:
                # Receive JPEG bytes from browser.
                data = await ws.receive_bytes()
                arr = np.frombuffer(data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                # Run detection.
                result = await loop.run_in_executor(None, detector.detect, frame_bgr)

                # Store latest frame + mask for inpainting.
                with state.lock:
                    state.latest_frame = frame_bgr.copy()
                    state.latest_mask = result.mask.copy()

                # Build mask preview.
                mask_preview = await loop.run_in_executor(
                    None, create_mask_preview, frame_bgr, result.mask
                )

                # Send back annotated + mask as base64 JPEG.
                resp = DetectionResponse(
                    detect=_encode_frame_jpeg(result.annotated_frame),
                    mask=_encode_frame_jpeg(mask_preview),
                    count=result.count,
                    label=result.label,
                )
                await ws.send_text(resp.model_dump_json())
            except Exception:
                logger.exception("Error processing detection frame")
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/inpaint")
async def ws_inpaint(ws: WebSocket):
    await ws.accept()
    orchestrator = InpaintOrchestrator(ws, state)
    try:
        while True:
            msg = await ws.receive_text()
            try:
                req = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_text(ErrorResponse(error="Invalid JSON").model_dump_json())
                continue

            if req.get("action") == "cancel":
                await orchestrator.handle_cancel()
            else:
                await orchestrator.handle_generate(req)

    except WebSocketDisconnect:
        await orchestrator.handle_cancel()


@app.get("/api/status")
async def api_status():
    loaded = {name: det.is_loaded for name, det in state.detectors.items()}
    with state.lock:
        inpaint = state.inpaint_status
        inpaint_detail = state.inpaint_status_detail
    return JSONResponse({
        "detectors": loaded,
        "detection": ModelStatus.READY if any(loaded.values()) else ModelStatus.LOADING,
        "inpaint": inpaint,
        "inpaint_detail": inpaint_detail,
    })


@app.post("/api/capture")
async def api_capture():
    """Freeze the current frame + mask for inpainting."""
    with state.lock:
        if state.latest_frame is None:
            return JSONResponse({"error": "No frame available"}, status_code=400)
        state.captured_frame = state.latest_frame.copy()
        mask = state.latest_mask
        state.captured_mask = mask.copy() if mask is not None else None
        count = int(mask.max() > 0) if mask is not None else 0

    # Build mask preview of the captured frame for the client.
    loop = asyncio.get_event_loop()
    if state.captured_mask is not None and state.captured_mask.max() > 0:
        mask_preview = await loop.run_in_executor(
            None, create_mask_preview, state.captured_frame, state.captured_mask
        )
    else:
        mask_preview = state.captured_frame.copy()

    return JSONResponse({
        "ok": True,
        "count": count,
        "mask": _encode_frame_jpeg(mask_preview),
    })


# Serve static files (frontend) — must be last (catch-all)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
