#!/bin/bash
# Entrypoint script for runtime model conversion
# Converts YOLO models to OpenVINO format on first startup if not already present

set -e

MODEL_DIR="${OPENVINO_MODEL_PATH:-/app/models}"
CONVERT_SCRIPT="/app/convert_models.py"

# Check if models need to be converted
check_models() {
    local models=("yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x")
    for model in "${models[@]}"; do
        if [ ! -f "$MODEL_DIR/${model}_openvino_model/${model}.xml" ]; then
            return 1
        fi
    done
    return 0
}

# Convert models if not present
if ! check_models; then
    echo "Models not found in $MODEL_DIR, converting..."
    mkdir -p "$MODEL_DIR"

    # Run conversion script
    python3 "$CONVERT_SCRIPT"

    # Move converted models to model directory
    for dir in *_openvino_model; do
        if [ -d "$dir" ]; then
            mv "$dir" "$MODEL_DIR/"
        fi
    done

    echo "Model conversion complete."
else
    echo "Models already present in $MODEL_DIR, skipping conversion."
fi

# Start the MCP server
exec python3 main.py "$@"
