#!/bin/bash
# Build all OpenVINO Object Detection MCP Server Docker images
#
# This script builds 4 container variants:
#   1. openvino-detection:ubuntu         - Ubuntu with pre-converted models (build-time conversion)
#   2. openvino-detection:rhel           - RHEL UBI with pre-converted models (build-time conversion)
#   3. openvino-detection:ubuntu-runtime - Ubuntu with runtime conversion (uses volume for models)
#   4. openvino-detection:rhel-runtime   - RHEL UBI with runtime conversion (uses volume for models)
#
# The "runtime" variants convert models on first startup and cache them in a volume.
# This results in faster builds but slower first startup.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
BUILD_ALL=true
BUILD_UBUNTU=false
BUILD_RHEL=false
BUILD_UBUNTU_RUNTIME=false
BUILD_RHEL_RUNTIME=false
NO_CACHE=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --all              Build all 4 images (default)"
    echo "  --ubuntu           Build only Ubuntu (build-time conversion)"
    echo "  --rhel             Build only RHEL (build-time conversion)"
    echo "  --ubuntu-runtime   Build only Ubuntu runtime (startup conversion)"
    echo "  --rhel-runtime     Build only RHEL runtime (startup conversion)"
    echo "  --no-cache         Build without using Docker cache"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Build all images"
    echo "  $0 --ubuntu --rhel          # Build only pre-converted images"
    echo "  $0 --ubuntu-runtime         # Build only Ubuntu runtime image"
    echo "  $0 --all --no-cache         # Rebuild all images from scratch"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            BUILD_ALL=true
            shift
            ;;
        --ubuntu)
            BUILD_ALL=false
            BUILD_UBUNTU=true
            shift
            ;;
        --rhel)
            BUILD_ALL=false
            BUILD_RHEL=true
            shift
            ;;
        --ubuntu-runtime)
            BUILD_ALL=false
            BUILD_UBUNTU_RUNTIME=true
            shift
            ;;
        --rhel-runtime)
            BUILD_ALL=false
            BUILD_RHEL_RUNTIME=true
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# If --all or no specific image selected, build all
if [ "$BUILD_ALL" = true ]; then
    BUILD_UBUNTU=true
    BUILD_RHEL=true
    BUILD_UBUNTU_RUNTIME=true
    BUILD_RHEL_RUNTIME=true
fi

log_info "Starting Docker image builds..."
echo ""

# Track build results
FAILED_BUILDS=()
SUCCESSFUL_BUILDS=()

build_image() {
    local dockerfile=$1
    local tag=$2
    local description=$3

    log_info "Building $description..."
    log_info "  Dockerfile: $dockerfile"
    log_info "  Tag: openvino-detection:$tag"
    echo ""

    if docker build $NO_CACHE -f "$dockerfile" -t "openvino-detection:$tag" .; then
        log_success "Built openvino-detection:$tag"
        SUCCESSFUL_BUILDS+=("openvino-detection:$tag")
    else
        log_error "Failed to build openvino-detection:$tag"
        FAILED_BUILDS+=("openvino-detection:$tag")
    fi
    echo ""
}

# Build Ubuntu (build-time conversion)
if [ "$BUILD_UBUNTU" = true ]; then
    build_image "Dockerfile.ubuntu" "ubuntu" "Ubuntu with pre-converted models"
fi

# Build RHEL (build-time conversion)
if [ "$BUILD_RHEL" = true ]; then
    build_image "Dockerfile.rhel" "rhel" "RHEL UBI with pre-converted models"
fi

# Build Ubuntu Runtime (startup conversion)
if [ "$BUILD_UBUNTU_RUNTIME" = true ]; then
    build_image "Dockerfile.ubuntu-runtime" "ubuntu-runtime" "Ubuntu with runtime conversion"
fi

# Build RHEL Runtime (startup conversion)
if [ "$BUILD_RHEL_RUNTIME" = true ]; then
    build_image "Dockerfile.rhel-runtime" "rhel-runtime" "RHEL UBI with runtime conversion"
fi

# Summary
echo ""
echo "========================================"
echo "Build Summary"
echo "========================================"

if [ ${#SUCCESSFUL_BUILDS[@]} -gt 0 ]; then
    log_success "Successfully built:"
    for img in "${SUCCESSFUL_BUILDS[@]}"; do
        echo "  - $img"
    done
fi

if [ ${#FAILED_BUILDS[@]} -gt 0 ]; then
    log_error "Failed to build:"
    for img in "${FAILED_BUILDS[@]}"; do
        echo "  - $img"
    done
    echo ""
    exit 1
fi

echo ""
log_info "To run a container:"
echo ""
echo "  # Pre-converted models (faster startup):"
echo "  docker run -d -p 8006:8006 openvino-detection:ubuntu"
echo ""
echo "  # Runtime conversion (needs volume for model cache):"
echo "  docker run -d -p 8006:8006 -v openvino-models:/app/models openvino-detection:ubuntu-runtime"
echo ""
log_info "To check container health:"
echo "  curl http://localhost:8006/mcp"
echo ""
