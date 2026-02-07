"""Stable Diffusion inpainting backend.

Moves SD inference into a forked child process so that cancellation is
instant (SIGTERM the child) and the parent's model state is never
corrupted.  The SD pipeline is loaded once in the parent and inherited
by children via Linux copy-on-write.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import multiprocessing
import signal
import sys
from typing import AsyncIterator, Optional

import torch
from PIL import Image

from server.inpainters.base import (
    BaseInpainter,
    InpaintDoneMsg,
    InpaintErrorMsg,
    InpaintMessage,
    InpaintProgressMsg,
)

logger = logging.getLogger(__name__)

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass  # Already set — can only be called once per process

# Loaded once in the parent process; children inherit via fork COW.
_SHARED_PIPE = None


def _worker_fn(child_conn, image_pil, mask_pil, prompt, num_steps, guidance_scale):
    """Run SD inference in a forked child process."""
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    try:
        pipe = _SHARED_PIPE

        def step_cb(pipe, step, timestep, callback_kwargs):
            child_conn.send({"type": "progress", "step": step + 1, "total": num_steps})
            return callback_kwargs

        result = pipe(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            callback_on_step_end=step_cb,
        )

        buf = io.BytesIO()
        result.images[0].save(buf, format="JPEG")
        encoded_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        child_conn.send({"type": "done", "image_b64": encoded_string})
    except Exception as e:
        logger.error("Worker error: %s", e, exc_info=True)
        child_conn.send({"type": "error", "message": "Generation failed"})
    finally:
        child_conn.close()


class StableDiffusionInpainter(BaseInpainter):
    """Process-based Stable Diffusion inpainting backend."""

    def __init__(
        self,
        model_id: str,
        device: str,
        num_steps: int,
        guidance_scale: float,
    ):
        self.model_id = model_id
        self.device = device
        self._num_steps = num_steps
        self.guidance_scale = guidance_scale
        self._process: Optional[multiprocessing.Process] = None
        self._parent_conn = None
        self._lock = asyncio.Lock()

    # --- BaseInpainter interface ---

    def load(self, status_callback=None):
        global _SHARED_PIPE
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline

        if status_callback:
            status_callback("downloading", "Downloading model weights (~4 GB)...")

        logger.info(
            "Loading inpainting model: %s (this may take a while on first run)...",
            self.model_id,
        )
        use_fp16 = self.device == "cuda"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            safety_checker=None,
        )

        if status_callback:
            status_callback("initializing", "Configuring pipeline...")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = pipe.to(self.device)

        if self.device == "cpu":
            pipe.enable_attention_slicing()

        _SHARED_PIPE = pipe
        logger.info("Inpainting model loaded. Device: %s", self.device)

    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
    ) -> AsyncIterator[InpaintMessage]:
        if self._process is not None and self._process.is_alive():
            yield InpaintErrorMsg(message="Generation already in progress")
            return

        parent_conn, child_conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=_worker_fn,
            args=(child_conn, image, mask, prompt, self._num_steps, self.guidance_scale),
        )
        self._process.start()
        child_conn.close()  # Parent only uses parent_conn
        self._parent_conn = parent_conn

        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    msg = await loop.run_in_executor(None, parent_conn.recv)
                except (EOFError, OSError):
                    async with self._lock:
                        proc = self._process
                    exitcode = proc.exitcode if proc else None
                    if exitcode is not None and exitcode == -15:
                        logger.info("Child terminated by SIGTERM (cancel)")
                    else:
                        logger.error("Child process crashed (exitcode=%s)", exitcode)
                        yield InpaintErrorMsg(message="Generation process crashed")
                    return

                if msg["type"] == "progress":
                    yield InpaintProgressMsg(step=msg["step"], total=msg["total"])
                elif msg["type"] == "done":
                    yield InpaintDoneMsg(image_b64=msg["image_b64"])
                    return
                elif msg["type"] == "error":
                    yield InpaintErrorMsg(message=msg["message"])
                    return
        finally:
            async with self._lock:
                self._reap_child(timeout=5)
                self._close_pipe()

    async def cancel(self) -> None:
        async with self._lock:
            if self._process is None:
                return
            self._process.terminate()
            self._reap_child(timeout=0.5)
            self._close_pipe()

    def shutdown(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        self._reap_child(timeout=0.5)
        self._close_pipe()

    @property
    def is_ready(self) -> bool:
        return _SHARED_PIPE is not None

    @property
    def total_steps(self) -> int:
        return self._num_steps

    # --- Internal helpers ---

    def _reap_child(self, timeout: float) -> None:
        """Wait for child process to exit, escalating to SIGKILL if needed."""
        if self._process is None:
            return
        pid = self._process.pid
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning("Child %d did not exit after %gs, sending SIGKILL", pid, timeout)
            self._process.kill()
            self._process.join(timeout=1.0)
        logger.info("Reaped child process %d (exitcode=%s)", pid, self._process.exitcode)
        self._process = None

    def _close_pipe(self) -> None:
        """Close the parent end of the pipe and clear the reference."""
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except OSError:
                pass
            self._parent_conn = None
