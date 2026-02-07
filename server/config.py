import os


class Settings:
    WEIGHTS_PATH: str = os.environ.get("WEIGHTS_PATH", "weights/yolov5n-0.5.pt")
    BISENET_WEIGHTS_PATH: str = os.environ.get(
        "BISENET_WEIGHTS_PATH", "weights/bisenet_face_parsing.pth"
    )
    YOLOV8_SEG_MODEL: str = os.environ.get("YOLOV8_SEG_MODEL", "yolov8n-seg.pt")
    DEFAULT_DETECTION_MODE: str = os.environ.get("DEFAULT_DETECTION_MODE", "face")
    IMG_SIZE: int = int(os.environ.get("IMG_SIZE", "320"))
    DEVICE: str = os.environ.get("DEVICE", "cpu")
    INPAINT_MODEL: str = os.environ.get(
        "INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"
    )
    INPAINT_STEPS: int = int(os.environ.get("INPAINT_STEPS", "8"))
    GUIDANCE_SCALE: float = float(os.environ.get("GUIDANCE_SCALE", "7.5"))


settings = Settings()
