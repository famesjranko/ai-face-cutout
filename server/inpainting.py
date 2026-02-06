import logging
from typing import Callable, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InpaintingEngine:
    def __init__(self, model_id: str, device: str, num_steps: int, guidance_scale: float):
        self.model_id = model_id
        self.device = device
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.pipe = None

    def load(self, status_callback=None):
        from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

        if status_callback:
            status_callback("downloading", "Downloading model weights (~4 GB)...")

        logger.info("Loading inpainting model: %s (this may take a while on first run)...", self.model_id)
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
        )

        if status_callback:
            status_callback("initializing", "Configuring pipeline...")

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(self.device)

        # Reduce memory on CPU
        if self.device == "cpu":
            self.pipe.enable_attention_slicing()

        logger.info("Inpainting model loaded.")

    def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("Inpainting model not loaded. Call load() first.")

        def step_callback(pipe, step, timestep, callback_kwargs):
            if progress_callback:
                progress_callback(step + 1, self.num_steps)
            return callback_kwargs

        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            callback_on_step_end=step_callback,
        )

        return result.images[0]
