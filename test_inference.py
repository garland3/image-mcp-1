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

    def mcp_call(self, method: str, params: dict) -> dict:
        """Make an MCP JSON-RPC call."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        self.log_verbose(f"Request: {json.dumps(payload, indent=2)[:500]}...")
        response = requests.post(self.mcp_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        self.log_verbose(f"Response: {json.dumps(result, indent=2)[:500]}...")
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
            if response.status_code == 200:
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

    def test_detect_objects_base64(self, image_base64: str = None) -> bool:
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

    def run_all_tests(self, image_path: str = None) -> bool:
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
            self.test_detect_objects_base64(image_base64)
        else:
            self.test_detect_objects_base64()
        
        # Summary
        print("\n" + "=" * 50)
        print(f"Test Summary: {self.passed} passed, {self.failed} failed")
        print("=" * 50 + "\n")
        
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test OpenVINO Object Detection MCP Server")
    parser.add_argument("--port", type=int, default=8006, help="Server port (default: 8006)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--image", type=str, help="Path to test image (optional)")
    args = parser.parse_args()
    
    tester = InferenceTest(port=args.port, verbose=args.verbose)
    success = tester.run_all_tests(image_path=args.image)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
