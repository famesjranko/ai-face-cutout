"""Pydantic models for WebSocket message schemas."""

from pydantic import BaseModel


# --- Detection ---

class DetectionResponse(BaseModel):
    detect: str
    mask: str
    count: int
    label: str


# --- Inpainting ---

class InpaintStarted(BaseModel):
    status: str = "started"
    total_steps: int


class InpaintProgress(BaseModel):
    status: str = "progress"
    step: int
    total_steps: int
    elapsed: float


class InpaintDone(BaseModel):
    status: str = "done"
    image: str
    elapsed: float


class InpaintCancelled(BaseModel):
    status: str = "cancelled"


# --- Shared ---

class ErrorResponse(BaseModel):
    error: str
