"""
Quick test script — run after starting the server.

Usage:
    1. Start the server:  uvicorn main:app --port 8000
    2. In another terminal: python test_api.py <path_to_image>

    Or without an image (uses a generated test image):
        python test_api.py
"""

import sys
import json
import requests
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health():
    print("Testing /health ...")
    r = requests.get(f"{API_URL}/health")
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    print("  ✅ Health check passed\n")


def test_root():
    print("Testing / ...")
    r = requests.get(API_URL)
    print(f"  Status: {r.status_code}")
    print(f"  Response: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    print("  ✅ Root check passed\n")


def test_caption(image_path: str = None):
    print("Testing /analyze-image ...")

    if image_path and Path(image_path).exists():
        print(f"  Using image: {image_path}")
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            r = requests.post(f"{API_URL}/analyze-image", files=files)
    else:
        # Generate a simple test image
        print("  Generating test image (red square)...")
        from PIL import Image
        import io

        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        files = {"file": ("test_red.jpg", buf, "image/jpeg")}
        r = requests.post(f"{API_URL}/analyze-image", files=files)

    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Response:")
    print(f"    filename:    {data.get('filename')}")
    print(f"    caption:     {data.get('caption')}")
    print(f"    translation: {data.get('translation')}")
    print(f"    insight:     {data.get('insight')}")

    assert r.status_code == 200
    assert "caption" in data
    assert "translation" in data
    assert len(data["caption"]) > 0
    assert len(data["translation"]) > 0
    print("  ✅ Caption pipeline passed\n")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    test_root()
    test_health()
    test_caption(image_path)

    print("=" * 50)
    print("All tests passed! Your backend is ready to deploy.")