import asyncio
import base64
import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.detectors import FaceBiSeNetDetector, YOLOv8SegDetector, BaseDetector
from server.masking import create_mask_preview, create_inpaint_inputs
from server.inpainting import InpaintingEngine, GenerationCancelled

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
    inpaint_status: str = "loading"
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
            state.inpaint_status = "ready"
            state.inpaint_status_detail = "Model ready"
    except Exception:
        with state.lock:
            state.inpaint_status = "error"
            state.inpaint_status_detail = "Failed to load model"
        logger.exception("Failed to load inpainting model")


def _prewarm_face_detector():
    """Pre-load the face detector in a background thread."""
    try:
        get_detector_sync("face")
    except Exception:
        logger.exception("Failed to pre-warm face detector")


def get_detector_sync(mode: str, target_classes=None) -> BaseDetector:
    """Get or create a detector by mode name (thread-safe, lazy-loading)."""
    with state.detector_lock:
        if mode == "face":
            if "face" not in state.detectors:
                det = FaceBiSeNetDetector(
                    yolo_weights=settings.WEIGHTS_PATH,
                    bisenet_weights=settings.BISENET_WEIGHTS_PATH,
                    device=settings.DEVICE,
                    img_size=settings.IMG_SIZE,
                )
                det.load()
                state.detectors["face"] = det
            return state.detectors["face"]

        elif mode == "object":
            if "object" not in state.detectors:
                det = YOLOv8SegDetector(
                    model_variant=settings.YOLOV8_SEG_MODEL,
                    device=settings.DEVICE,
                    target_classes=target_classes or [0],
                )
                det.load()
                state.detectors["object"] = det
            else:
                # Update target classes if the detector already exists.
                if target_classes is not None:
                    state.detectors["object"].set_target_classes(target_classes)
            return state.detectors["object"]

        else:
            raise ValueError(f"Unknown detection mode: {mode}")


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
        await ws.send_text(json.dumps({"error": f"Failed to load detector: {exc}"}))
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
                resp = {
                    "detect": _encode_frame_jpeg(result.annotated_frame),
                    "mask": _encode_frame_jpeg(mask_preview),
                    "count": result.count,
                    "label": result.label,
                }
                await ws.send_text(json.dumps(resp))
            except Exception:
                logger.exception("Error processing detection frame")
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/inpaint")
async def ws_inpaint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                req = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            if req.get("action") == "cancel":
                with state.lock:
                    engine = state.inpaint_engine
                if engine is not None:
                    engine.cancel()
                continue

            prompt = req.get("prompt", "")

            if not prompt:
                await ws.send_text(json.dumps({"error": "No prompt provided"}))
                continue

            with state.lock:
                engine = state.inpaint_engine
            if engine is None or engine.pipe is None:
                await ws.send_text(json.dumps({"error": "Inpainting model still loading, please wait..."}))
                continue

            # Use the explicitly captured frame.
            with state.lock:
                frame = state.captured_frame
                mask = state.captured_mask

            if frame is None:
                await ws.send_text(json.dumps({"error": "No frame captured. Click Capture first."}))
                continue

            if mask is None or mask.max() == 0:
                await ws.send_text(json.dumps({"error": "No detection in captured frame"}))
                continue

            # Prepare inpainting inputs.
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None, create_inpaint_inputs, frame, mask
            )
            if inputs is None:
                await ws.send_text(json.dumps({"error": "Failed to create mask"}))
                continue

            image_pil, mask_pil = inputs

            await ws.send_text(json.dumps({
                "status": "started",
                "total_steps": settings.INPAINT_STEPS,
            }))

            start_time = time.time()

            # Progress callback sends updates over WebSocket.
            progress_queue: asyncio.Queue = asyncio.Queue()

            def on_progress(step: int, total: int):
                elapsed = time.time() - start_time
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"status": "progress", "step": step, "total_steps": total, "elapsed": round(elapsed, 1)},
                )

            # Run generation in executor.
            gen_task = loop.run_in_executor(
                None, engine.generate, image_pil, mask_pil, prompt, on_progress
            )

            # Listen for cancel messages while sending progress updates.
            cancelled = False
            ws_recv = asyncio.ensure_future(ws.receive_text())

            while not gen_task.done():
                progress_get = asyncio.ensure_future(progress_queue.get())

                done, pending = await asyncio.wait(
                    [gen_task, ws_recv, progress_get],
                    timeout=0.5,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if progress_get in pending:
                    progress_get.cancel()

                for task in done:
                    if task is ws_recv:
                        try:
                            cancel_msg = json.loads(task.result())
                            if cancel_msg.get("action") == "cancel":
                                engine.cancel()
                                cancelled = True
                        except Exception:
                            pass
                        # Only create a new receiver when the old one completed
                        if not gen_task.done():
                            ws_recv = asyncio.ensure_future(ws.receive_text())
                    elif task is progress_get:
                        try:
                            await ws.send_text(json.dumps(task.result()))
                        except Exception:
                            pass

            # Clean up the pending ws receiver
            if not ws_recv.done():
                ws_recv.cancel()
                try:
                    await ws_recv
                except (asyncio.CancelledError, Exception):
                    pass

            try:
                result_image = await gen_task
            except GenerationCancelled:
                cancelled = True

            if cancelled:
                await ws.send_text(json.dumps({"status": "cancelled"}))
                continue

            # Drain remaining progress messages.
            while not progress_queue.empty():
                progress_msg = progress_queue.get_nowait()
                await ws.send_text(json.dumps(progress_msg))

            # Encode result as base64 JPEG.
            buf = io.BytesIO()
            result_image.save(buf, format="JPEG", quality=90)
            result_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            elapsed = round(time.time() - start_time, 1)
            await ws.send_text(json.dumps({
                "status": "done",
                "image": result_b64,
                "elapsed": elapsed,
            }))

    except WebSocketDisconnect:
        # If disconnected during generation, cancel it.
        with state.lock:
            engine = state.inpaint_engine
        if engine is not None:
            engine.cancel()


@app.get("/api/status")
async def api_status():
    loaded = {name: det.is_loaded for name, det in state.detectors.items()}
    with state.lock:
        inpaint = state.inpaint_status
        inpaint_detail = state.inpaint_status_detail
    return JSONResponse({
        "detectors": loaded,
        "detection": "ready" if any(loaded.values()) else "loading",
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
