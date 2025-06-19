#!/usr/bin/env python3
import argparse
import base64
import requests
import numpy as np
import cv2
import os

API_URL = "http://127.0.0.1:8000/embed"

def get_embedding(image_path: str, api_url: str) -> np.ndarray:
    """
    Gets the face embedding for a pre-aligned image by calling the API.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read the image to get its dimensions and content
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w, _ = image.shape

    # Since the image is pre-aligned/cropped, the bbox is the whole image
    bbox = {"x": 0, "y": 0, "w": w, "h": h}

    # Encode image to base64
    _, img_encoded = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    # Prepare the request payload
    payload = {
        "image_base64": img_base64,
        "bbox": bbox,
        "landmarks": None # No landmarks needed as it's pre-aligned
    }

    try:
        print(f"Requesting embedding for {os.path.basename(image_path)}...")
        response = requests.post(api_url, json=payload, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        vector = data.get("vector")
        
        if not vector:
            raise ValueError("API response did not contain an embedding vector.")
            
        print("Embedding received successfully.")
        return np.array(vector)

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

def main():
    parser = argparse.ArgumentParser(
        description="Compare two pre-aligned face images using the FaceEmbed API.\n"
                    "Example:\n"
                    "python scripts/compare_aligned_faces.py debug_images/aligned_one.jpg debug_images/aligned_two.jpg",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("image1", help="Path to the first pre-aligned face image.")
    parser.add_argument("image2", help="Path to the second pre-aligned face image.")
    parser.add_argument(
        "--api-url", 
        default=API_URL, 
        help=f"URL of the /embed endpoint.\n(default: {API_URL})"
    )

    args = parser.parse_args()

    print(f"Comparing '{os.path.basename(args.image1)}' and '{os.path.basename(args.image2)}'...")

    # Get embeddings for both images
    embedding1 = get_embedding(args.image1, args.api_url)
    if embedding1 is None:
        return
        
    embedding2 = get_embedding(args.image2, args.api_url)
    if embedding2 is None:
        return

    # Calculate and print similarity
    similarity = cosine_similarity(embedding1, embedding2)
    
    print("-" * 30)
    print(f"Cosine Similarity: {similarity:.4f}")
    
    # Simple thresholding for demonstration
    if similarity > 0.6:
        print("Result: The faces are likely the SAME person.")
    elif similarity > 0.4:
        print("Result: The faces are somewhat similar but might be DIFFERENT people.")
    else:
        print("Result: The faces are likely DIFFERENT people.")
        
    print("\nNote: Thresholds are illustrative. Optimal values depend on the model and use case.")
    print("Typical thresholds for ArcFace models:")
    print(" - High Security (low False Accept Rate): ~0.68")
    print(" - Medium Security: ~0.5")
    print(" - Low Security (low False Reject Rate): ~0.4")


if __name__ == "__main__":
    main() 