# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Notes

- **Container Runtime**: This machine has `podman` available (Docker-compatible CLI)

## Container Build Testing

**IMPORTANT**: When testing or verifying Docker container builds, use the `container-builder` agent via the Task tool instead of running docker build commands directly. This agent runs `create_containers.sh` and provides concise feedback on build success/failure.

## Container Inference Testing

**IMPORTANT**: After building containers, use the `container-inference-tester` agent via the Task tool to validate that containers are running correctly and inference is working. This agent starts containers, runs inference tests, and reports pass/fail results.

## Project Overview

OpenVINO Object Detection MCP Server - A FastMCP-based HTTP server providing YOLOv11 object detection with Intel OpenVINO optimization. Detects 80 COCO object classes with hardware-accelerated inference.

## Common Commands

```bash
# Local development
pip install -r requirements.txt
python main.py                              # HTTP server on 127.0.0.1:8006/mcp
python main.py --host 0.0.0.0 --port 8080   # Custom host/port
python main.py --stdio                       # STDIO transport
python main.py --sse --port 8006            # SSE transport

# Validation (no dependencies required)
python validate_structure.py

# Docker - use create_containers.sh to build all images
./create_containers.sh                      # Build all 4 container variants
./create_containers.sh --ubuntu             # Build only Ubuntu pre-converted
./create_containers.sh --ubuntu-runtime     # Build only Ubuntu runtime conversion
docker-compose up -d
docker-compose logs -f

# Kubernetes/Helm
helm install openvino-server ./helm/openvino-server --namespace mcp --create-namespace
helm upgrade openvino-server ./helm/openvino-server --namespace mcp
kubectl logs -n mcp deployment/openvino-server

# Model conversion (requires requirements-build.txt dependencies)
pip install -r requirements-build.txt
python convert_models.py

# Inference testing (after containers are running)
./test_inference.sh                         # Quick bash-based test
./test_inference.sh 8080                    # Test on custom port
python test_inference.py                    # Test with test.png, output to test_output.png
python test_inference.py --suffix ubuntu    # Output to test_output_ubuntu.png
python test_inference.py --suffix rhel      # Output to test_output_rhel.png
python test_inference.py --verbose          # With debug output
python test_inference.py --image foo.jpg -o result.png  # Custom input/output
```

## Architecture

### Core Components

- **main.py**: FastMCP server implementation with 4 MCP tools and 1 resource
  - `detect_objects`: File-based image detection
  - `detect_objects_base64`: Base64-encoded image detection
  - `list_available_models`: Available YOLO model variants
  - `get_class_labels`: 80 COCO class labels
  - `detection://info`: Server information resource

### Model Loading and Caching

Models are loaded via `_get_model()` which uses a global `_model_cache` dictionary keyed by `{model_name}_{device}`. The `_ensure_model_downloaded()` function:
1. Checks `OPENVINO_MODEL_PATH` env var (default: `/app/models`) for pre-converted models (Docker)
2. Falls back to runtime conversion using ultralytics if available (local development)
3. Stores converted models in `tempfile.gettempdir()/openvino_models/`

### Image Processing Pipeline

1. `_load_image()` or base64 decode
2. `_preprocess_image()`: Resize with letterboxing to 640x640, normalize to [0,1], CHW format
3. OpenVINO inference
4. `_postprocess_detections()`: Extract boxes, apply confidence threshold, NMS, scale to original coordinates
5. `_draw_detections()`: Overlay bounding boxes and labels
6. Return base64-encoded PNG overlay

### Docker Variants

Four Dockerfile variants are available:
- **Dockerfile.ubuntu** / **Dockerfile.rhel**: Multi-stage builds with pre-converted models (build-time conversion)
- **Dockerfile.ubuntu-runtime** / **Dockerfile.rhel-runtime**: Runtime conversion on startup (requires volume mount)

The pre-converted variants use 3 stages:
1. **converter**: Installs ultralytics/torch, converts all 5 YOLO models to OpenVINO format
2. **builder**: Installs runtime dependencies only
3. **runtime**: Minimal image with pre-converted models (30-40% smaller)

The runtime variants convert models on first startup and cache them in `/app/models` (mount a volume to persist).

## Key Implementation Details

- **Default port**: 8006
- **MCP endpoint**: `/mcp` for HTTP transport
- **Model variants**: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x (nano to extra-large)
- **Input size**: 640x640 with letterbox padding (gray value 114)
- **NMS**: Uses `cv2.dnn.NMSBoxes` with default IoU threshold 0.45
- **Confidence threshold**: Default 0.25
- **Device selection**: "AUTO" (default), "CPU", "GPU" via OpenVINO

## Dependencies

- **Runtime** (`requirements.txt`): openvino, opencv-python-headless, numpy, Pillow, fastmcp
- **Build-time** (`requirements-build.txt`): ultralytics, torch, torchvision, openvino (for model conversion)

## Test Files

- **test.png**: Default test image for inference testing
- **test_output*.png**: Generated output images with detection overlays (gitignored)
