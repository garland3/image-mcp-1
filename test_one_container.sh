#!/bin/bash
# Test a single container variant

IMAGE=$1
NAME=$2
PORT=8006

if [ -z "$IMAGE" ] || [ -z "$NAME" ]; then
    echo "Usage: $0 <image> <name>"
    exit 1
fi

echo "========================================"
echo "Testing: $NAME"
echo "Image: $IMAGE"
echo "========================================"

# Clean up
podman rm -f "openvino-test-$NAME" 2>/dev/null || true

# Start container
echo "[INFO] Starting container..."
if ! podman run -d --name "openvino-test-$NAME" -p "$PORT:$PORT" "$IMAGE"; then
    echo "[FAIL] Container failed to start"
    exit 1
fi

# Wait for server
echo "[INFO] Waiting for server to be ready..."
sleep 5

for i in {1..30}; do
    if curl -s http://localhost:$PORT/mcp | grep -q "jsonrpc"; then
        echo "[PASS] Server ready after $i attempts"
        break
    fi
    sleep 2
done

# Run test
echo "[INFO] Running inference test..."
if python3 /home/garlan/git/image-mcp-1/test_inference.py --port $PORT; then
    echo "[PASS] $NAME: All tests passed"
    RESULT=0
else
    echo "[FAIL] $NAME: Tests failed"
    RESULT=1
fi

# Cleanup
podman rm -f "openvino-test-$NAME" 2>/dev/null || true

exit $RESULT
