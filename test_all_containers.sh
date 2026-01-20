#!/bin/bash
# Test all 4 OpenVINO container variants with podman
# Tests inference on each container individually

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CONTAINER_PORT=8006
HOST_PORT=8006
WARMUP_TIME=5
MAX_WAIT=60

log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_header() { echo -e "${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}"; }

# Container variants to test
declare -A CONTAINERS
CONTAINERS["ubuntu"]="localhost/openvino-detection:ubuntu"
CONTAINERS["rhel"]="localhost/openvino-detection:rhel"
CONTAINERS["ubuntu-runtime"]="localhost/openvino-detection:ubuntu-runtime"
CONTAINERS["rhel-runtime"]="localhost/openvino-detection:rhel-runtime"

PASSED=0
FAILED=0
declare -A TEST_RESULTS

# Function to test a single container
test_container() {
    local name=$1
    local image=$2
    local container_name="openvino-test-${name}"

    log_header "Testing: $name ($image)"

    # Clean up any existing container with this name
    log_info "Cleaning up any existing container..."
    podman rm -f "$container_name" 2>/dev/null || true

    # Start the container
    log_info "Starting container $container_name..."
    if ! podman run -d \
        --name "$container_name" \
        -p "${HOST_PORT}:${CONTAINER_PORT}" \
        "$image"; then
        log_fail "Failed to start container $name"
        TEST_RESULTS[$name]="FAIL: Container failed to start"
        ((FAILED++))
        return 1
    fi

    log_pass "Container started successfully"

    # Wait for container to be ready
    log_info "Waiting ${WARMUP_TIME}s for container initialization..."
    sleep $WARMUP_TIME

    # Wait for server to respond (with timeout)
    log_info "Waiting for MCP server to be ready..."
    local elapsed=0
    local server_ready=false

    while [ $elapsed -lt $MAX_WAIT ]; do
        # Just check if server is responding (even with 406 is fine, means it's up)
        if curl -s "http://localhost:${HOST_PORT}/mcp" | grep -q "jsonrpc"; then
            server_ready=true
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        echo -n "."
    done
    echo ""

    if [ "$server_ready" = false ]; then
        log_fail "Server did not become ready within ${MAX_WAIT}s"
        log_info "Container logs:"
        podman logs "$container_name" | tail -30
        podman rm -f "$container_name" 2>/dev/null || true
        TEST_RESULTS[$name]="FAIL: Server did not start in time"
        ((FAILED++))
        return 1
    fi

    log_pass "MCP server is ready after ${elapsed}s"

    # Run inference test
    log_info "Running inference test with Python test suite..."
    if python3 /home/garlan/git/image-mcp-1/test_inference.py --port $HOST_PORT; then
        log_pass "Inference test PASSED for $name"
        TEST_RESULTS[$name]="PASS"
        ((PASSED++))
    else
        log_fail "Inference test FAILED for $name"
        log_info "Container logs:"
        podman logs "$container_name" | tail -30
        TEST_RESULTS[$name]="FAIL: Inference test failed"
        ((FAILED++))
    fi

    # Cleanup
    log_info "Stopping and removing container..."
    podman rm -f "$container_name" 2>/dev/null || true

    echo ""
}

# Main test loop
log_header "OpenVINO Container Inference Test Suite"
echo "Testing 4 container variants with podman"
echo ""

for name in ubuntu rhel ubuntu-runtime rhel-runtime; do
    test_container "$name" "${CONTAINERS[$name]}"
    echo ""
done

# Final summary
log_header "TEST SUMMARY"
echo ""
for name in ubuntu rhel ubuntu-runtime rhel-runtime; do
    result="${TEST_RESULTS[$name]}"
    if [[ $result == PASS* ]]; then
        log_pass "$name: $result"
    else
        log_fail "$name: $result"
    fi
done

echo ""
echo -e "${BLUE}Total:${NC} $PASSED passed, $FAILED failed out of 4 containers"
echo ""

if [ $FAILED -eq 0 ]; then
    log_header "ALL TESTS PASSED!"
    exit 0
else
    log_header "SOME TESTS FAILED"
    exit 1
fi
