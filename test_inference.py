#!/usr/bin/env python3
"""
Detailed inference test for OpenVINO Object Detection MCP Server.

Usage:
    python test_inference.py [--port PORT] [--verbose] [--image PATH]
"""

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
from PIL import Image


class InferenceTest:
    def __init__(self, port: int = 8006, verbose: bool = False):
        self.base_url = f"http://localhost:{port}"
        self.mcp_url = f"{self.base_url}/mcp"
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.session_id = None

    def log(self, msg: str):
        print(f"[TEST] {msg}")

    def log_verbose(self, msg: str):
        if self.verbose:
            print(f"[DEBUG] {msg}")

    def log_pass(self, msg: str):
        print(f"\033[92m[PASS]\033[0m {msg}")
        self.passed += 1

    def log_fail(self, msg: str):
        print(f"\033[91m[FAIL]\033[0m {msg}")
        self.failed += 1

    def _initialize_session(self):
        """Initialize MCP session."""
        if not self.session_id:
            # Get session ID
            response = requests.get(self.mcp_url)
            self.session_id = response.headers.get('mcp-session-id')
            self.log_verbose(f"Got session ID: {self.session_id}")

            # Initialize the session
            init_payload = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "openvino-test",
                        "version": "1.0.0"
                    }
                }
            }
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": self.session_id
            }
            init_response = requests.post(self.mcp_url, json=init_payload, headers=headers, timeout=30)
            init_response.raise_for_status()
            # Parse SSE response if needed
            if 'text/event-stream' in init_response.headers.get('content-type', ''):
                for line in init_response.text.split('\n'):
                    if line.startswith('data: '):
                        init_result = json.loads(line[6:])
                        self.log_verbose(f"Session initialized: {init_result}")
                        break
            else:
                self.log_verbose(f"Session initialized: {init_response.json()}")

    def mcp_call(self, method: str, params: dict) -> dict:
        """Make an MCP JSON-RPC call."""
        # Ensure session is initialized
        if not self.session_id:
            self._initialize_session()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        self.log_verbose(f"Request: {json.dumps(payload, indent=2)[:500]}...")
        # FastMCP HTTP transport requires both content types in Accept header
        # and a session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id
        }
        response = requests.post(self.mcp_url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        self.log_verbose(f"Response status: {response.status_code}")
        self.log_verbose(f"Response content-type: {response.headers.get('content-type')}")

        # Parse SSE format if needed
        if 'text/event-stream' in response.headers.get('content-type', ''):
            # Extract JSON from SSE format: "event: message\ndata: {...}"
            for line in response.text.split('\n'):
                if line.startswith('data: '):
                    result = json.loads(line[6:])  # Remove "data: " prefix
                    self.log_verbose(f"Response JSON: {json.dumps(result, indent=2)[:500]}...")
                    return result
            raise ValueError("No data found in SSE response")
        else:
            result = response.json()
            self.log_verbose(f"Response JSON: {json.dumps(result, indent=2)[:500]}...")
            return result

    def create_test_image(self, width: int = 640, height: int = 480) -> str:
        """Create a test image and return its base64 encoding."""
        # Create image with colored rectangles that might be detected
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add some colored shapes
        img[50:200, 50:200] = [255, 0, 0]    # Red square
        img[100:300, 300:500] = [0, 255, 0]  # Green rectangle
        img[250:400, 100:250] = [0, 0, 255]  # Blue rectangle
        
        # Save to temporary file and encode
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            Image.fromarray(img).save(f.name)
            with open(f.name, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')

    def load_image_base64(self, path: str) -> str:
        """Load an image file and return its base64 encoding."""
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def test_server_health(self) -> bool:
        """Test if server is responding."""
        self.log("Testing server health...")
        try:
            response = requests.get(self.mcp_url, timeout=10)
            # FastMCP HTTP transport returns 406 for GET without SSE headers
            # Any response (200, 406, etc) means server is up
            if response.status_code in [200, 406]:
                self.log_pass("Server is healthy")
                return True
            else:
                self.log_fail(f"Server returned status {response.status_code}")
                return False
        except Exception as e:
            self.log_fail(f"Server not responding: {e}")
            return False

    def test_list_models(self) -> bool:
        """Test list_available_models tool."""
        self.log("Testing list_available_models...")
        try:
            result = self.mcp_call("tools/call", {
                "name": "list_available_models",
                "arguments": {}
            })
            
            if "error" in result:
                self.log_fail(f"Error: {result['error']}")
                return False
            
            content = result.get("result", {}).get("content", [])
            if content and "yolo11n" in str(content):
                self.log_pass("list_available_models returned expected models")
                return True
            else:
                self.log_fail("list_available_models did not return expected models")
                return False
        except Exception as e:
            self.log_fail(f"Exception: {e}")
            return False

    def test_get_class_labels(self) -> bool:
        """Test get_class_labels tool."""
        self.log("Testing get_class_labels...")
        try:
            result = self.mcp_call("tools/call", {
                "name": "get_class_labels",
                "arguments": {}
            })
            
            if "error" in result:
                self.log_fail(f"Error: {result['error']}")
                return False
            
            content = result.get("result", {}).get("content", [])
            if content and "person" in str(content):
                self.log_pass("get_class_labels returned COCO classes")
                return True
            else:
                self.log_fail("get_class_labels did not return expected classes")
                return False
        except Exception as e:
            self.log_fail(f"Exception: {e}")
            return False

    def test_detect_objects_base64(self, image_base64: str = None, output_path: str = None) -> bool:
        """Test detect_objects_base64 tool."""
        self.log("Testing detect_objects_base64...")
        try:
            if image_base64 is None:
                self.log("Creating test image...")
                image_base64 = self.create_test_image()

            result = self.mcp_call("tools/call", {
                "name": "detect_objects_base64",
                "arguments": {
                    "image_base64": image_base64,
                    "model_name": "yolo11n",
                    "confidence_threshold": 0.1
                }
            })

            if "error" in result:
                self.log_fail(f"Error: {result['error']}")
                return False

            content = result.get("result", {}).get("content", [])
            if not content:
                self.log_fail("No content in response")
                return False

            # Parse the text content
            text_content = content[0].get("text", "{}")
            data = json.loads(text_content)

            if "results" in data:
                detection_count = data["results"].get("detection_count", 0)
                has_overlay = "overlay_base64" in data["results"]
                inference_time = data.get("meta_data", {}).get("inference_time_ms", "unknown")

                # Save overlay image if available
                if has_overlay and output_path:
                    overlay_base64 = data["results"]["overlay_base64"]
                    overlay_bytes = base64.b64decode(overlay_base64)
                    with open(output_path, 'wb') as f:
                        f.write(overlay_bytes)
                    self.log(f"Saved overlay image to: {output_path}")

                # Print detections
                detections = data["results"].get("detections", [])
                if detections:
                    self.log("Detections:")
                    for det in detections:
                        self.log(f"  - {det['class_name']}: {det['confidence']:.2f} at [{det['bbox']['x1']:.0f}, {det['bbox']['y1']:.0f}, {det['bbox']['x2']:.0f}, {det['bbox']['y2']:.0f}]")

                self.log_pass(f"Detection succeeded: {detection_count} objects, overlay={has_overlay}, inference={inference_time}ms")
                return True
            elif "error" in data.get("results", {}):
                self.log_fail(f"Detection error: {data['results']['error']}")
                return False
            else:
                self.log_fail(f"Unexpected response format: {text_content[:200]}")
                return False
        except Exception as e:
            self.log_fail(f"Exception: {e}")
            return False

    def run_all_tests(self, image_path: str = None, output_path: str = None) -> bool:
        """Run all tests and return overall success."""
        print("\n" + "=" * 50)
        print("OpenVINO Object Detection MCP Server - Inference Tests")
        print("=" * 50 + "\n")

        # Server health is required
        if not self.test_server_health():
            print("\nServer not available, aborting tests.")
            return False

        # Run all other tests
        self.test_list_models()
        self.test_get_class_labels()

        # Detection test with optional custom image
        if image_path:
            self.log(f"Using custom image: {image_path}")
            image_base64 = self.load_image_base64(image_path)
            self.test_detect_objects_base64(image_base64, output_path)
        else:
            self.test_detect_objects_base64(output_path=output_path)

        # Summary
        print("\n" + "=" * 50)
        print(f"Test Summary: {self.passed} passed, {self.failed} failed")
        print("=" * 50 + "\n")

        return self.failed == 0


def main():
    # Get script directory for default paths
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Test OpenVINO Object Detection MCP Server")
    parser.add_argument("--port", type=int, default=8006, help="Server port (default: 8006)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--image", type=str, default=str(script_dir / "test.png"),
                        help="Path to test image (default: test.png)")
    parser.add_argument("--output", "-o", type=str,
                        help="Path to save overlay image (default: test_output_<suffix>.png)")
    parser.add_argument("--suffix", "-s", type=str, default="",
                        help="Suffix to add to output filename (e.g., 'ubuntu', 'rhel')")
    args = parser.parse_args()

    # Build output path with suffix
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{args.suffix}" if args.suffix else ""
        output_path = str(script_dir / f"test_output{suffix}.png")

    tester = InferenceTest(port=args.port, verbose=args.verbose)
    success = tester.run_all_tests(image_path=args.image, output_path=output_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
