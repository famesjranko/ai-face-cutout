from enum import Enum


class DetectionMode(str, Enum):
    FACE = "face"
    OBJECT = "object"


class ModelStatus(str, Enum):
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
