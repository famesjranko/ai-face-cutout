"""Encapsulates the inpainting generation lifecycle.

Backend-agnostic: consumes the async generator produced by any
BaseInpainter implementation rather than managing pipes/processes
directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket
from pydantic import BaseModel

from server.inpainters.base import InpaintDoneMsg, InpaintErrorMsg, InpaintProgressMsg
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
        """Cancel any in-progress generation.  Safe to call at any time."""
        inpainter = self._state.inpainter
        if inpainter is None:
            return
        try:
            await inpainter.cancel()
        except Exception:
            logger.exception("Error during cancel")

    async def handle_generate(self, req: dict):
        """Validate inputs and run the generation lifecycle."""
        prompt = req.get("prompt", "")

        if not prompt:
            await self._send_model(ErrorResponse(error="No prompt provided"))
            return

        inpainter = self._state.inpainter
        if inpainter is None or not inpainter.is_ready:
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

        loop = asyncio.get_running_loop()
        inputs = await loop.run_in_executor(None, create_inpaint_inputs, frame, mask)
        if inputs is None:
            await self._send_model(ErrorResponse(error="Failed to create mask"))
            return

        image_pil, mask_pil = inputs
        await self._run_generation(inpainter, image_pil, mask_pil, prompt)

    _GENERATION_TIMEOUT = 600  # 10 minutes

    async def _run_generation(self, inpainter, image_pil, mask_pil, prompt):
        """Execute generation with progress streaming and cancel support."""
        await self._send_model(InpaintStarted(total_steps=inpainter.total_steps))

        start_time = time.time()
        gen = inpainter.generate(image_pil, mask_pil, prompt)

        gen_task = asyncio.ensure_future(gen.__anext__())
        ws_recv = asyncio.ensure_future(self._ws.receive_text())

        try:
            while True:
                # Timeout check
                if time.time() - start_time > self._GENERATION_TIMEOUT:
                    logger.warning("Generation timed out after %ds", self._GENERATION_TIMEOUT)
                    await inpainter.cancel()
                    await self._send_model(ErrorResponse(error="Generation timed out"))
                    break

                done, _ = await asyncio.wait(
                    [gen_task, ws_recv],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if gen_task in done:
                    try:
                        msg = gen_task.result()
                    except StopAsyncIteration:
                        break

                    if isinstance(msg, InpaintProgressMsg):
                        elapsed = time.time() - start_time
                        await self._send_model(
                            InpaintProgress(
                                step=msg.step,
                                total_steps=msg.total,
                                elapsed=round(elapsed, 1),
                            )
                        )
                        gen_task = asyncio.ensure_future(gen.__anext__())
                    elif isinstance(msg, InpaintDoneMsg):
                        elapsed = time.time() - start_time
                        await self._send_model(
                            InpaintDone(
                                image=msg.image_b64,
                                elapsed=round(elapsed, 1),
                            )
                        )
                        break
                    elif isinstance(msg, InpaintErrorMsg):
                        await self._send_model(ErrorResponse(error=msg.message))
                        break

                if ws_recv in done:
                    try:
                        raw = ws_recv.result()
                        cancel_msg = json.loads(raw)
                        if cancel_msg.get("action") == "cancel":
                            await inpainter.cancel()
                            await self._send_model(InpaintCancelled())
                            break
                    except Exception:
                        pass
                    ws_recv = asyncio.ensure_future(self._ws.receive_text())
        finally:
            # Clean up the pending ws receiver
            if not ws_recv.done():
                ws_recv.cancel()
                try:
                    await ws_recv
                except (asyncio.CancelledError, Exception):
                    pass
            # Clean up pending generator task
            if not gen_task.done():
                gen_task.cancel()
                try:
                    await gen_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Close the async generator
            try:
                await gen.aclose()
            except Exception:
                pass

    async def _send_model(self, model: BaseModel):
        await self._ws.send_text(model.model_dump_json())
