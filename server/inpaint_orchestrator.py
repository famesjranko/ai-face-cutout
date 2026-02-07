"""Encapsulates the inpainting generation lifecycle."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time

from fastapi import WebSocket
from pydantic import BaseModel

from server.config import settings
from server.inpainting import GenerationCancelled
from server.masking import create_inpaint_inputs
from server.schemas import (
    ErrorResponse,
    InpaintCancelled,
    InpaintDone,
    InpaintProgress,
    InpaintStarted,
)

logger = logging.getLogger(__name__)


class InpaintOrchestrator:
    """Manages validation, generation, progress streaming, and cancellation."""

    def __init__(self, ws: WebSocket, state):
        self._ws = ws
        self._state = state

    async def handle_cancel(self):
        """Cancel any in-progress generation."""
        with self._state.lock:
            engine = self._state.inpaint_engine
        if engine is not None:
            engine.cancel()

    async def handle_generate(self, req: dict):
        """Validate inputs and run the generation lifecycle."""
        prompt = req.get("prompt", "")

        if not prompt:
            await self._send_model(ErrorResponse(error="No prompt provided"))
            return

        with self._state.lock:
            engine = self._state.inpaint_engine
        if engine is None or engine.pipe is None:
            await self._send_model(ErrorResponse(error="Inpainting model still loading, please wait..."))
            return

        with self._state.lock:
            frame = self._state.captured_frame
            mask = self._state.captured_mask

        if frame is None:
            await self._send_model(ErrorResponse(error="No frame captured. Click Capture first."))
            return

        if mask is None or mask.max() == 0:
            await self._send_model(ErrorResponse(error="No detection in captured frame"))
            return

        loop = asyncio.get_event_loop()
        inputs = await loop.run_in_executor(None, create_inpaint_inputs, frame, mask)
        if inputs is None:
            await self._send_model(ErrorResponse(error="Failed to create mask"))
            return

        image_pil, mask_pil = inputs
        await self._run_generation(engine, image_pil, mask_pil, prompt, loop)

    async def _run_generation(self, engine, image_pil, mask_pil, prompt, loop):
        """Execute generation with progress streaming and cancel support."""
        await self._send_model(InpaintStarted(total_steps=settings.INPAINT_STEPS))

        start_time = time.time()
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_progress(step: int, total: int):
            elapsed = time.time() - start_time
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                InpaintProgress(step=step, total_steps=total, elapsed=round(elapsed, 1)),
            )

        gen_task = loop.run_in_executor(
            None, engine.generate, image_pil, mask_pil, prompt, on_progress
        )

        cancelled = False
        ws_recv = asyncio.ensure_future(self._ws.receive_text())

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
                    if not gen_task.done():
                        ws_recv = asyncio.ensure_future(self._ws.receive_text())
                elif task is progress_get:
                    try:
                        await self._send_model(task.result())
                    except Exception:
                        pass

        # Clean up the pending ws receiver.
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
            await self._send_model(InpaintCancelled())
            return

        # Drain remaining progress messages.
        while not progress_queue.empty():
            progress_msg = progress_queue.get_nowait()
            await self._send_model(progress_msg)

        # Encode result as base64 JPEG.
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG", quality=90)
        result_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        elapsed = round(time.time() - start_time, 1)
        await self._send_model(InpaintDone(image=result_b64, elapsed=elapsed))

    async def _send_model(self, model: BaseModel):
        await self._ws.send_text(model.model_dump_json())
