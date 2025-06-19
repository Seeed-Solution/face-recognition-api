#!/usr/bin/env python3
"""
测试Hailo硬件的API请求
"""

import base64
import cv2
import numpy as np
import requests
import pytest

# --- Configuration ---
BASE_URL = "http://localhost:8000"
IMAGE_PATH = "face.jpg"  # Create a dummy image or use a real one
COLLECTION_NAME = "my_pytest_collection"
USER_ID = "user_pytest_001"

def image_to_base64(img_path):
    """Encodes an image to base64"""
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at '{img_path}'. Creating a dummy file.")
        # Create a dummy black image if not found
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

@pytest.mark.dependency()
def test_health():
    """Tests the /health endpoint"""
    print("\n--- Testing /health ---")
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    response.raise_for_status()
    print(f"✅ Health check successful: {response.json()}")
    assert response.status_code == 200

@pytest.mark.dependency(depends=["test_health"])
def test_detect_and_embed():
    """Tests the /detect_and_embed endpoint and returns a vector"""
    print("\n--- Testing /detect_and_embed ---")
    image_b64 = image_to_base64(IMAGE_PATH)
    payload = {"image_base64": image_b64}
    response = requests.post(f"{BASE_URL}/detect_and_embed", json=payload, timeout=15)
    response.raise_for_status()
    results = response.json()
    print(f"✅ /detect_and_embed successful. Found {len(results)} faces.")
    assert response.status_code == 200
    if results:
        # Pass the vector to the next test via pytest.shared
        pytest.vector = results[0]["embedding"]["vector"]
    else:
        pytest.vector = None

@pytest.mark.dependency(depends=["test_detect_and_embed"])
def test_vector_db_flow():
    """Tests the full vector DB flow: add, search, delete"""
    vector = getattr(pytest, "vector", None)
    if not vector:
        pytest.skip("Skipping DB test because no vector was obtained from previous step.")

    print("\n--- Testing Vector DB Flow ---")

    # 1. Add vector
    print(f"1. Adding vector for user '{USER_ID}' to collection '{COLLECTION_NAME}'...")
    payload = {"collection": COLLECTION_NAME, "user_id": USER_ID, "vector": vector}
    response = requests.post(f"{BASE_URL}/vectors/add", json=payload, timeout=5)
    response.raise_for_status()
    print(f"   ✅ Add successful: {response.json()}")
    assert response.status_code == 200

    # 2. Search for the vector
    print("2. Searching for the added vector...")
    payload = {"collection": COLLECTION_NAME, "vector": vector, "threshold": 0.9}
    response = requests.post(f"{BASE_URL}/vectors/search", json=payload, timeout=5)
    response.raise_for_status()
    results = response.json()
    assert results.get("status") == "found"
    assert results.get("results")
    print(f"   ✅ Search successful: Found match with similarity {results['results'][0]['similarity']:.4f}")

    # 3. Delete the vector
    print(f"3. Deleting vector for user '{USER_ID}'...")
    payload = {"collection": COLLECTION_NAME, "user_id": USER_ID}
    response = requests.post(f"{BASE_URL}/vectors/delete", json=payload, timeout=5)
    response.raise_for_status()
    print(f"   ✅ Delete successful: {response.json()}")
    assert response.json()["deleted_count"] > 0


if __name__ == "__main__":
    test_health()
    test_detect_and_embed()
    test_vector_db_flow()
    print("\n--- Test script finished ---") 