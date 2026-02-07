"""Base class and message types for inpainting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional

if TYPE_CHECKING:
    from PIL import Image


# --- Progress message dataclasses (internal protocol, not WebSocket schemas) ---

@dataclass
class InpaintProgressMsg:
    """Per-step progress update from the backend."""
    step: int
    total: int


@dataclass
class InpaintDoneMsg:
    """Final result — base64-encoded JPEG image."""
    image_b64: str


@dataclass
class InpaintErrorMsg:
    """Error during generation."""
    message: str


# Union type for yield values
InpaintMessage = InpaintProgressMsg | InpaintDoneMsg | InpaintErrorMsg


class BaseInpainter(ABC):
    """Common interface for all inpainting backends.

    Key difference from detectors: inpainting is streaming (progress updates
    over time), so ``generate()`` is an async generator that yields typed
    message dataclasses.
    """

    @abstractmethod
    def load(self, status_callback: Optional[Callable] = None) -> None:
        """Load model weights / initialise the backend.

        *status_callback(status, detail)* is called to report loading
        progress (e.g. downloading weights).
        """

    @abstractmethod
    async def generate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
    ) -> AsyncIterator[InpaintMessage]:
        """Run inpainting and yield progress messages.

        Yields ``InpaintProgressMsg`` for each step, then exactly one of
        ``InpaintDoneMsg`` or ``InpaintErrorMsg`` before returning.
        """
        # Needs `yield` to be recognised as an async generator by Python.
        yield  # pragma: no cover

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel any in-progress generation."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources on application shutdown."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier, e.g. ``"stable_diffusion"``."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the backend is loaded and ready to generate."""

    @property
    @abstractmethod
    def total_steps(self) -> int:
        """Number of inference steps (1 for single-shot API backends)."""

    @staticmethod
    @abstractmethod
    def backend_id() -> str:
        """Registry key matching :class:`InpaintBackend` enum value."""
