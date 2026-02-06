import asyncio
import base64
import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from server.config import settings
from server.detection import load_model, detect_frame
from server.masking import create_mask_preview, create_inpaint_inputs
from server.inpainting import InpaintingEngine

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    model: Optional[torch.nn.Module] = None
    inpaint_engine: Optional[InpaintingEngine] = None
    latest_frame: Optional[np.ndarray] = None
    latest_detections: Optional[torch.Tensor] = None
    captured_frame: Optional[np.ndarray] = None
    captured_detections: Optional[torch.Tensor] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    inpaint_status: str = "loading"
    inpaint_status_detail: str = "Starting up..."


state = AppState()


def _load_inpaint_model():
    """Load SD inpainting model in background thread."""
    def on_status(status, detail):
        state.inpaint_status = status
        state.inpaint_status_detail = detail

    try:
        state.inpaint_engine = InpaintingEngine(
            model_id=settings.INPAINT_MODEL,
            device=settings.DEVICE,
            num_steps=settings.INPAINT_STEPS,
            guidance_scale=settings.GUIDANCE_SCALE,
        )
        state.inpaint_engine.load(status_callback=on_status)
        state.inpaint_status = "ready"
        state.inpaint_status_detail = "Model ready"
    except Exception:
        state.inpaint_status = "error"
        state.inpaint_status_detail = "Failed to load model"
        logger.exception("Failed to load inpainting model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load YOLOv5 model (fast, <2s)
    logger.info("Loading YOLOv5 face model...")
    state.model = load_model(settings.WEIGHTS_PATH, settings.DEVICE)
    logger.info("YOLOv5 model loaded.")

    # Load SD inpainting model in background so server starts immediately
    logger.info("Loading inpainting model in background (may take minutes on first run)...")
    bg = threading.Thread(target=_load_inpaint_model, daemon=True)
    bg.start()

    yield  # App is running

    # Shutdown
    state.model = None
    state.inpaint_engine = None


app = FastAPI(lifespan=lifespan)


def _encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive JPEG bytes from browser
            data = await ws.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue

            # Run detection
            loop = asyncio.get_event_loop()
            annotated, det = await loop.run_in_executor(
                None, detect_frame, state.model, frame_bgr, settings.IMG_SIZE, settings.DEVICE
            )

            # Store latest frame + detections for inpainting
            with state.lock:
                state.latest_frame = frame_bgr.copy()
                state.latest_detections = det.clone() if len(det) else det

            # Build mask preview
            mask_preview = await loop.run_in_executor(
                None, create_mask_preview, frame_bgr, det
            )

            # Send back annotated + mask as base64 JPEG
            resp = {
                "detect": _encode_frame_jpeg(annotated),
                "mask": _encode_frame_jpeg(mask_preview),
                "faces": len(det),
            }
            await ws.send_text(json.dumps(resp))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/inpaint")
async def ws_inpaint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            req = json.loads(msg)
            prompt = req.get("prompt", "")

            if not prompt:
                await ws.send_text(json.dumps({"error": "No prompt provided"}))
                continue

            if state.inpaint_engine is None or state.inpaint_engine.pipe is None:
                await ws.send_text(json.dumps({"error": "Inpainting model still loading, please wait..."}))
                continue

            # Use the explicitly captured frame
            with state.lock:
                frame = state.captured_frame
                det = state.captured_detections

            if frame is None:
                await ws.send_text(json.dumps({"error": "No frame captured. Click Capture first."}))
                continue

            if det is None or len(det) == 0:
                await ws.send_text(json.dumps({"error": "No face detected in captured frame"}))
                continue

            # Prepare inpainting inputs
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None, create_inpaint_inputs, frame, det
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

            # Progress callback sends updates over WebSocket
            # We need to use a thread-safe queue since the callback runs in the executor
            progress_queue: asyncio.Queue = asyncio.Queue()

            def on_progress(step: int, total: int):
                elapsed = time.time() - start_time
                # Put into queue (thread-safe via asyncio)
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {"status": "progress", "step": step, "total_steps": total, "elapsed": round(elapsed, 1)},
                )

            # Run generation in executor
            gen_task = loop.run_in_executor(
                None, state.inpaint_engine.generate, image_pil, mask_pil, prompt, on_progress
            )

            # Send progress updates while waiting for generation
            while not gen_task.done():
                try:
                    progress_msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    await ws.send_text(json.dumps(progress_msg))
                except asyncio.TimeoutError:
                    pass

            result_image = await gen_task

            # Drain remaining progress messages
            while not progress_queue.empty():
                progress_msg = progress_queue.get_nowait()
                await ws.send_text(json.dumps(progress_msg))

            # Encode result as base64 JPEG
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
        pass


@app.get("/api/status")
async def api_status():
    return JSONResponse({
        "yolo": "ready" if state.model is not None else "loading",
        "inpaint": state.inpaint_status,
        "inpaint_detail": state.inpaint_status_detail,
    })


@app.post("/api/capture")
async def api_capture():
    """Freeze the current frame + detections for inpainting."""
    with state.lock:
        if state.latest_frame is None:
            return JSONResponse({"error": "No frame available"}, status_code=400)
        state.captured_frame = state.latest_frame.copy()
        det = state.latest_detections
        state.captured_detections = (
            det.clone() if det is not None and len(det) else det
        )
        faces = len(det) if det is not None else 0

    # Build mask preview of the captured frame for the client
    loop = asyncio.get_event_loop()
    if state.captured_detections is not None and len(state.captured_detections):
        mask_preview = await loop.run_in_executor(
            None, create_mask_preview, state.captured_frame, state.captured_detections
        )
    else:
        mask_preview = state.captured_frame.copy()

    return JSONResponse({
        "ok": True,
        "faces": faces,
        "mask": _encode_frame_jpeg(mask_preview),
    })


# Serve static files (frontend) — must be last (catch-all)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
