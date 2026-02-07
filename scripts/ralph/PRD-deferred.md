# Deferred Codebase Health Improvements

Branch: `feature/codebase-health`
Quality gate: `python -m py_compile` (no venv — deps not installed locally)

## Tasks

### 1. [x] Detector registry pattern
Replace hardcoded if/elif detector creation in `get_detector_sync()` (server/app.py:78-109) with a registry that auto-discovers detectors from `server/detectors/`. Adding a third detector should require ZERO changes to app.py.

**Implementation:**
- Add a `registry` dict to `server/detectors/__init__.py` mapping `DetectionMode` → detector class
- Each detector subclass registers itself (either via decorator or explicit registration in `__init__.py`)
- Refactor `get_detector_sync()` to look up the registry instead of hardcoding `FaceBiSeNetDetector` / `YOLOv8SegDetector`
- app.py should import from `server.detectors` only — no direct detector class imports
- Preserve existing lazy-loading + `detector_lock` thread safety
- Preserve `set_target_classes()` behavior for object detector

**Files:** `server/detectors/__init__.py`, `server/app.py`, possibly `server/detectors/base.py`
**Quality:** `python -m py_compile server/app.py && python -m py_compile server/detectors/__init__.py && python -m py_compile server/detectors/base.py`

### 2. [x] WebSocket message schemas with Pydantic
Replace ad-hoc dict construction for WebSocket messages with Pydantic models. Both detection and inpainting WebSocket handlers build response dicts inline — add typed models so the frontend/backend contract is explicit.

**Implementation:**
- Create `server/schemas.py` with Pydantic BaseModel classes for all WebSocket messages:
  - Detection: `DetectionResponse(detect, mask, count, label)`
  - Inpainting: `InpaintStarted(status, total_steps)`, `InpaintProgress(status, step, total_steps, elapsed)`, `InpaintDone(status, image, elapsed)`, `InpaintCancelled(status)`, `ErrorResponse(error)`
- Update `ws_detect` in app.py to use `DetectionResponse`
- Update `InpaintOrchestrator` in `server/inpaint_orchestrator.py` to use the inpainting schemas
- Use `.model_dump()` instead of manual dict construction, then `json.dumps()` as before
- Do NOT change the JSON keys — frontend expects exact field names (`detect`, `mask`, `count`, `label`, `status`, `step`, `total_steps`, `elapsed`, `image`, `error`)

**Files:** `server/schemas.py` (new), `server/app.py`, `server/inpaint_orchestrator.py`
**Quality:** `python -m py_compile server/schemas.py && python -m py_compile server/app.py && python -m py_compile server/inpaint_orchestrator.py`

### 3. [x] Frontend state machine refactor
Refactor `static/app.js` from 27+ loose globals and nested callbacks into a clean state machine. The IIFE currently mixes UI state, WebSocket management, and rendering in one flat scope.

**Implementation:**
- Introduce a simple state object (plain JS, no framework) that holds all app state: `{ camera, detection, inpainting, ui }`
- Group related functions into namespaces or modules (Camera, Detection, Inpainting, UI)
- Replace scattered `var` globals with state object properties
- Keep the IIFE pattern (no modules/bundler), but structure the code with clear sections
- Preserve all existing functionality — detection stream, capture, inpainting with progress/cancel, mode switching
- Do NOT add any npm dependencies or build tools

**Files:** `static/app.js`
**Quality:** Manual review (no compiler for JS). Check that the file is valid JS: `node --check static/app.js` (if node available) or just ensure no syntax errors.

## Context

This is the AI Face Cutout project — a FastAPI backend with WebSocket streaming for real-time face/object detection and inpainting. The codebase just went through 7 health fixes (inference locks, error handling, InpaintOrchestrator extraction, enums, YOLOv5 pruning). These 3 tasks are the "deferred" items from the review.

**Key patterns already in place:**
- `BaseDetector` ABC with `DetectionResult` dataclass (server/detectors/base.py)
- `DetectionMode` and `ModelStatus` enums (server/enums.py)
- `InpaintOrchestrator` class (server/inpaint_orchestrator.py)
- Thread-safe detector creation via `state.detector_lock`
- `_inference_lock` on each detector instance for thread-safe inference

**Gotchas:**
- Python deps NOT installed locally — use `py_compile` not imports
- Frontend is vanilla JS, no build tools, served as static files
- WebSocket JSON keys must NOT change (frontend depends on exact names)
- `pydantic` IS in the project deps (FastAPI depends on it) but can't be imported locally
