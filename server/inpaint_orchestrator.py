"""Encapsulates the inpainting generation lifecycle.

Uses a forked child process for SD inference so that cancellation is
instant (kill the child) rather than waiting for the current diffusion
step to finish.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import WebSocket
from pydantic import BaseModel

from server.config import settings
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
        worker = self._state.inpaint_worker
        if worker is not None:
            worker.cancel()

    async def handle_generate(self, req: dict):
        """Validate inputs and run the generation lifecycle."""
        prompt = req.get("prompt", "")

        if not prompt:
            await self._send_model(ErrorResponse(error="No prompt provided"))
            return

        worker = self._state.inpaint_worker
        if worker is None or not worker.is_ready:
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
        await self._run_generation(worker, image_pil, mask_pil, prompt, loop)

    async def _run_generation(self, worker, image_pil, mask_pil, prompt, loop):
        """Execute generation with progress streaming and cancel support."""
        await self._send_model(InpaintStarted(total_steps=settings.INPAINT_STEPS))

        start_time = time.time()

        try:
            parent_conn = worker.spawn(image_pil, mask_pil, prompt)
        except RuntimeError as exc:
            await self._send_model(ErrorResponse(error=str(exc)))
            return

        ws_recv = asyncio.ensure_future(self._ws.receive_text())
        pipe_task = None

        try:
            while True:
                pipe_task = loop.run_in_executor(None, parent_conn.recv)

                done, pending = await asyncio.wait(
                    [pipe_task, ws_recv],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if pipe_task in done:
                    try:
                        msg = pipe_task.result()
                    except EOFError:
                        # Child died — check exit code to distinguish cancel vs crash
                        exitcode = worker._process.exitcode if worker._process else None
                        if exitcode is not None and exitcode == -15:
                            # SIGTERM — this was a cancel
                            pass
                        else:
                            await self._send_model(
                                ErrorResponse(error="Generation process crashed")
                            )
                        break

                    if msg["type"] == "progress":
                        elapsed = time.time() - start_time
                        await self._send_model(
                            InpaintProgress(
                                step=msg["step"],
                                total_steps=msg["total"],
                                elapsed=round(elapsed, 1),
                            )
                        )
                    elif msg["type"] == "done":
                        elapsed = time.time() - start_time
                        await self._send_model(
                            InpaintDone(
                                image=msg["image_b64"],
                                elapsed=round(elapsed, 1),
                            )
                        )
                        break
                    elif msg["type"] == "error":
                        await self._send_model(ErrorResponse(error=msg["message"]))
                        break

                if ws_recv in done:
                    try:
                        raw = ws_recv.result()
                        cancel_msg = json.loads(raw)
                        if cancel_msg.get("action") == "cancel":
                            worker.cancel()
                            await self._send_model(InpaintCancelled())
                            break
                    except Exception:
                        pass
                    # Start listening for the next WS message
                    ws_recv = asyncio.ensure_future(self._ws.receive_text())

                if pipe_task in pending:
                    pipe_task.cancel()
        finally:
            # Clean up the pending ws receiver
            if not ws_recv.done():
                ws_recv.cancel()
                try:
                    await ws_recv
                except (asyncio.CancelledError, Exception):
                    pass
            # Clean up pending pipe task
            if pipe_task is not None and not pipe_task.done():
                pipe_task.cancel()
                try:
                    await pipe_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Join child process and close pipe
            worker.cleanup_child()

    async def _send_model(self, model: BaseModel):
        await self._ws.send_text(model.model_dump_json())
