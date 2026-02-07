# AI Photo Standee

AI-powered photo standee — like the cutout boards at amusement parks where you stick your face through a hole in a painted scene, but powered by generative AI.

Your webcam detects and segments your face (or any object) in real time, then Stable Diffusion inpaints the scene around it based on a text prompt. No API keys needed — everything runs locally.

## How It Works

1. **Start your webcam** — choose between face segmentation or object/person detection mode
2. **Real-time detection** — face mode uses YOLOv5-Face + BiSeNet for pixel-level face masks; object mode uses YOLOv8-seg instance segmentation
3. **Capture a frame** — freezes the current detection and mask for inpainting
4. **Enter a prompt** — describe the scene (e.g. "astronaut on the moon", "pirate on a ship")
5. **Generate** — Stable Diffusion inpaints everything outside the detected region based on the prompt
6. **Download** — save the generated image

## Quick Start

```bash
docker compose up --build
```

Open **http://localhost:8000** in your browser.

The inpainting model (~4 GB) downloads automatically on first run and is cached in a Docker volume for subsequent starts.

## Architecture

```
Browser                              Docker Container
+--------------------------+         +--------------------------------+
| Webcam (getUserMedia)    | WS/JPEG | FastAPI + Uvicorn              |
| Mode selector (face/obj) | ------> | /ws/detect -> detectors/       |
| Detection + Mask display | <------ |   face: YOLOv5 + BiSeNet       |
|                          |         |   object: YOLOv8-seg           |
| Prompt + Generate        | WS/JSON |                                |
| Progress bar + Cancel    | ------> | /ws/inpaint -> forked SD child |
| Generated image display  | <------ |   progress + result            |
+--------------------------+         +--------------------------------+
```

- **Detection**: Multi-model system via `server/detectors/` package
  - *Face mode*: YOLOv5-Face bounding boxes + BiSeNet face parsing for pixel-level segmentation masks
  - *Object mode*: YOLOv8-seg instance segmentation (person, car, dog, etc.)
- **Inpainting**: Stable Diffusion (`runwayml/stable-diffusion-inpainting`) with DPMSolver at 8 steps, run in a forked child process for instant cancellation
- **Aspect ratio**: Webcam frames are resized preserving aspect ratio and padded with white (which becomes extra inpaint area for SD)
- **CPU-only**: No GPU required (generation takes ~30-90s on CPU)

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `WEIGHTS_PATH` | `weights/yolov5n-0.5.pt` | YOLOv5-Face model weights |
| `BISENET_WEIGHTS_PATH` | `weights/bisenet_face_parsing.pth` | BiSeNet face parsing weights (auto-downloaded) |
| `YOLOV8_SEG_MODEL` | `yolov8n-seg.pt` | YOLOv8 segmentation model variant |
| `DEFAULT_DETECTION_MODE` | `face` | Default detection mode (`face` or `object`) |
| `IMG_SIZE` | `320` | YOLOv5 input resolution |
| `INPAINT_MODEL` | `runwayml/stable-diffusion-inpainting` | HuggingFace model ID |
| `INPAINT_STEPS` | `8` | Inference steps (more = better quality, slower) |
| `GUIDANCE_SCALE` | `7.5` | Prompt adherence strength |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

## Project Structure

```
server/
  app.py                # FastAPI app, WebSocket endpoints, lifespan
  config.py             # Settings from environment variables
  detectors/            # Multi-model detection package
    __init__.py          #   registry + factory (create_detector)
    base.py              #   BaseDetector ABC + DetectionResult
    face_bisenet.py      #   YOLOv5-Face bbox + BiSeNet face parsing
    bisenet_model.py     #   BiSeNet network architecture
    yolov8_seg.py        #   YOLOv8-seg instance segmentation
  detection.py          # YOLOv5 face detection helpers
  yolov5_compat.py      # Consolidated YOLOv5 model code + torch.load shims
  masking.py            # Mask preview + SD inpaint input preparation
  inpaint_orchestrator.py  # Generation lifecycle, progress streaming, cancellation
  inpaint_worker.py     # Forked child process for SD inference
  enums.py              # DetectionMode, ModelStatus enums
  schemas.py            # Pydantic models for WebSocket messages
  run.py                # Entry point
static/
  index.html            # UI layout
  style.css             # Dark theme styling
  app.js                # WebSocket client, webcam, canvas, UI logic
weights/                # YOLOv5 face detection weights
Dockerfile
docker-compose.yml
requirements-web.txt    # Python dependencies
```

## Acknowledgements

- [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) for face detection
- [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) for face parsing segmentation
- [YOLOv8](https://github.com/ultralytics/ultralytics) for instance segmentation
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) by RunwayML
- Original concept inspired by amusement park face cutout boards
