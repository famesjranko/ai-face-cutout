from server.enums import InpaintBackend
from server.inpainters.base import (
    BaseInpainter,
    InpaintDoneMsg,
    InpaintErrorMsg,
    InpaintMessage,
    InpaintProgressMsg,
)
from server.inpainters.stable_diffusion import StableDiffusionInpainter

__all__ = [
    "BaseInpainter",
    "InpaintDoneMsg",
    "InpaintErrorMsg",
    "InpaintMessage",
    "InpaintProgressMsg",
    "registry",
    "create_inpainter",
]

# Maps InpaintBackend value -> inpainter class.
# Adding a new backend only requires:
#   1. Implementing BaseInpainter with backend_id()
#   2. Adding a _SETTINGS_MAP entry below
#   3. Importing the class here
registry = {
    cls.backend_id(): cls
    for cls in [StableDiffusionInpainter]
}

# Maps InpaintBackend value -> callable(settings) returning constructor kwargs.
_SETTINGS_MAP = {
    InpaintBackend.STABLE_DIFFUSION: lambda s: dict(
        model_id=s.INPAINT_MODEL,
        device=s.DEVICE,
        num_steps=s.INPAINT_STEPS,
        guidance_scale=s.GUIDANCE_SCALE,
    ),
}


def create_inpainter(backend: str, settings=None, **kwargs) -> BaseInpainter:
    """Instantiate an inpainter by backend name.

    If *settings* is provided, constructor kwargs are derived from the
    settings object via ``_SETTINGS_MAP``.  Extra *kwargs* are merged in
    (and override settings-derived values).
    """
    cls = registry.get(backend)
    if cls is None:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown inpaint backend: {backend!r} (available: {available})"
        )
    if settings is not None and backend in _SETTINGS_MAP:
        base_kwargs = _SETTINGS_MAP[backend](settings)
        base_kwargs.update(kwargs)
        return cls(**base_kwargs)
    return cls(**kwargs)
