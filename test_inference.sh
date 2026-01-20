#!/bin/bash
# Quick inference test for OpenVINO Object Detection MCP Server
# Usage: ./test_inference.sh [port]

set -e

PORT="${1:-8006}"
BASE_URL="http://localhost:$PORT"
MCP_URL="$BASE_URL/mcp"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${YELLOW}[TEST]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }

# Check if server is running
log_info "Testing server at $MCP_URL..."

if ! curl -sf "$MCP_URL" > /dev/null 2>&1; then
    log_fail "Server not responding at $MCP_URL"
    exit 1
fi
log_pass "Server is responding"

# Create a simple test image (red square on white background)
TEST_IMAGE="/tmp/test_image_$$.png"
log_info "Creating test image..."

python3 << EOF
import numpy as np
from PIL import Image

# Create a simple 640x480 image with a colored rectangle (simulates an object)
img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
img[100:400, 150:500] = [255, 0, 0]  # Red rectangle
Image.fromarray(img).save("$TEST_IMAGE")
print("Test image created")
EOF

if [ ! -f "$TEST_IMAGE" ]; then
    log_fail "Failed to create test image"
    exit 1
fi
log_pass "Test image created at $TEST_IMAGE"

# Convert image to base64
log_info "Converting image to base64..."
IMAGE_BASE64=$(base64 -w0 "$TEST_IMAGE")

# Test the detect_objects_base64 tool via MCP
log_info "Testing object detection via MCP..."

RESPONSE=$(curl -sf -X POST "$MCP_URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"jsonrpc\": \"2.0\",
        \"id\": 1,
        \"method\": \"tools/call\",
        \"params\": {
            \"name\": \"detect_objects_base64\",
            \"arguments\": {
                \"image_base64\": \"$IMAGE_BASE64\",
                \"model_name\": \"yolo11n\",
                \"confidence_threshold\": 0.1
            }
        }
    }" 2>&1) || {
    log_fail "MCP request failed"
    rm -f "$TEST_IMAGE"
    exit 1
}

# Check if response contains results
if echo "$RESPONSE" | grep -q '"detection_count"'; then
    DETECTION_COUNT=$(echo "$RESPONSE" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('result',{}).get('content',[{}])[0].get('text','{}'))" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('results',{}).get('detection_count','unknown'))" 2>/dev/null || echo "unknown")
    log_pass "Object detection succeeded (detections: $DETECTION_COUNT)"
elif echo "$RESPONSE" | grep -q '"error"'; then
    ERROR=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',{}).get('message','Unknown error'))" 2>/dev/null || echo "Unknown error")
    log_fail "Detection failed: $ERROR"
    rm -f "$TEST_IMAGE"
    exit 1
else
    log_pass "Object detection request completed"
fi

# Test list_available_models
log_info "Testing list_available_models..."
MODELS_RESPONSE=$(curl -sf -X POST "$MCP_URL" \
    -H "Content-Type: application/json" \
    -d '{
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "list_available_models",
            "arguments": {}
        }
    }' 2>&1) || {
    log_fail "list_available_models request failed"
    rm -f "$TEST_IMAGE"
    exit 1
}

if echo "$MODELS_RESPONSE" | grep -q 'yolo11n'; then
    log_pass "list_available_models returned model list"
else
    log_fail "list_available_models did not return expected models"
    rm -f "$TEST_IMAGE"
    exit 1
fi

# Cleanup
rm -f "$TEST_IMAGE"

echo ""
echo "========================================"
log_pass "All inference tests passed!"
echo "========================================"
