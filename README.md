# AI Face Cutout

AI-powered face cutout board — like the ones at amusement parks where you stick your face through a hole in a painted scene, but powered by generative AI.

Your webcam detects your face with YOLOv5, then Stable Diffusion inpaints the scene around it based on a text prompt. No API keys needed — everything runs locally.

## How It Works

1. **Start your webcam** — real-time face detection with bounding boxes and landmarks
2. **Capture a frame** — freezes the current detection for inpainting
3. **Enter a prompt** — describe the scene (e.g. "astronaut on the moon", "pirate on a ship")
4. **Generate** — Stable Diffusion inpaints everything outside your face based on the prompt
5. **Download** — save the generated image

## Quick Start

```bash
docker compose up --build
```

Open **http://localhost:8000** in your browser.

The inpainting model (~4 GB) downloads automatically on first run and is cached in a Docker volume for subsequent starts.

## Architecture

```
Browser                              Docker Container
+--------------------------+         +---------------------------+
| Webcam (getUserMedia)    | WS/JPEG | FastAPI + Uvicorn         |
| Canvas frame capture     | ------> | /ws/detect  -> YOLOv5     |
| Detection + Mask display | <------ |   annotated frame + mask  |
|                          |         |                           |
| Prompt + Generate        | WS/JSON | /ws/inpaint -> SD Pipeline|
| Progress bar             | ------> |   runs in thread pool     |
| Generated image display  | <------ |   progress + result       |
+--------------------------+         +---------------------------+
```

- **Detection**: YOLOv5 face model with real-time WebSocket streaming
- **Inpainting**: Stable Diffusion (`runwayml/stable-diffusion-inpainting`) with DPMSolver at 8 steps
- **Aspect ratio**: Webcam frames are resized preserving aspect ratio and padded with white (which becomes extra inpaint area for SD)
- **CPU-only**: No GPU required (generation takes ~30-90s on CPU)

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `WEIGHTS_PATH` | `weights/yolov5n-face.pt` | YOLOv5 face model weights |
| `INPAINT_MODEL` | `runwayml/stable-diffusion-inpainting` | HuggingFace model ID |
| `INPAINT_STEPS` | `8` | Inference steps (more = better quality, slower) |
| `GUIDANCE_SCALE` | `7.5` | Prompt adherence strength |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

## Project Structure

```
server/
  app.py          # FastAPI app, WebSocket endpoints, lifespan
  config.py       # Settings from environment variables
  detection.py    # YOLOv5 face detection (extracted from original)
  masking.py      # Mask creation for inpainting pipeline
  inpainting.py   # SD inpainting engine wrapper
  run.py          # Entry point
static/
  index.html      # UI layout
  style.css       # Dark theme styling
  app.js          # WebSocket client, webcam, canvas, UI logic
models/           # YOLOv5 model architecture code
utils/            # YOLOv5 utility functions
weights/          # YOLOv5 face detection weights
Dockerfile
docker-compose.yml
```

## Acknowledgements

- [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) for face detection
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) by RunwayML
- Original concept inspired by amusement park face cutout boards
