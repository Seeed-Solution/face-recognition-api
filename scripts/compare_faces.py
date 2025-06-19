#!/usr/bin/env python3
"""
Compare Face Embeddings Script

This script loads two preprocessed face images, generates their embedding
vectors using the Hailo-8 model, and computes the cosine similarity
between them to measure how similar they are.

Usage:
    python scripts/compare_faces.py <image1_path> <image2_path>

Example:
    python scripts/compare_faces.py debug_images/preprocessed_1.jpg debug_images/preprocessed_2.jpg
"""

import os
import sys
import queue
import threading
import cv2
import numpy as np

# Adjust the path to import from the parent directory's `src` folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.face_embed_api.utils import HailoAsyncInference

# --- Configuration ---
MODEL_PATH = os.getenv(
    'FACE_RECOGNITION_HEF', 
    'models/arcface_mobilefacenet.hef'
)
# Ensure the path is relative to the project root
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', MODEL_PATH
)


def get_embedding(hailo_inference: HailoAsyncInference, image: np.ndarray) -> np.ndarray:
    """
    Performs inference on a single image and returns the embedding vector.
    """
    # The HailoAsyncInference expects a batch.
    # The `send_original_frame` is True, so we send a tuple of 
    # (original_frame_batch, preprocessed_frame_batch)
    # Here, our image is already preprocessed, so we send it for both.
    hailo_inference.input_queue.put(([image], [image]))

    try:
        # Retrieve result from the output queue
        _, result = hailo_inference.output_queue.get(timeout=5.0)

        # Process result to get a 1D vector
        if isinstance(result, dict):
            embedding = list(result.values())[0]
        else:
            embedding = result
        
        embedding = np.array(embedding).flatten()

        # L2 normalization (critical for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)

    except queue.Empty:
        print("Error: Inference timed out.")
        return np.array([])

def main():
    """Main function to compare two faces."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_faces.py <image1_path> <image2_path>")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]

    if not os.path.exists(image_path1) or not os.path.exists(image_path2):
        print(f"Error: One or both image paths do not exist.")
        print(f"  - Checked: {image_path1}")
        print(f"  - Checked: {image_path2}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    # --- Initialize Hailo Inference ---
    input_q = queue.Queue(maxsize=10)
    output_q = queue.Queue(maxsize=10)

    hailo_inference = HailoAsyncInference(
        hef_path=MODEL_PATH,
        input_queue=input_q,
        output_queue=output_q,
        send_original_frame=True  # As in app.py
    )

    inference_thread = threading.Thread(target=hailo_inference.run, daemon=True)
    inference_thread.start()
    print(f"Successfully loaded model: {os.path.basename(MODEL_PATH)}")

    # --- Load and Process Images ---
    print(f"\nLoading images...")
    print(f"  - Image 1: {image_path1}")
    print(f"  - Image 2: {image_path2}")
    
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Error: Failed to read one or both images with OpenCV.")
        sys.exit(1)
        
    print("Generating embeddings...")
    vector1 = get_embedding(hailo_inference, img1)
    vector2 = get_embedding(hailo_inference, img2)

    # --- Stop Inference Thread ---
    input_q.put(None)
    inference_thread.join(timeout=5.0)

    if vector1.size == 0 or vector2.size == 0:
        print("\nCould not generate embeddings. Exiting.")
        sys.exit(1)
        
    # --- Compare Vectors ---
    cosine_similarity = np.dot(vector1, vector2)
    
    print("\n--- Results ---")
    print(f"Vector 1 (first 8 dims): {vector1[:8]}")
    print(f"Vector 2 (first 8 dims): {vector2[:8]}")
    print(f"\nCosine Similarity: {cosine_similarity:.4f}")
    
    # Interpretation
    if cosine_similarity > 0.6:
        print("Verdict: The faces are LIKELY the same person.")
    elif cosine_similarity > 0.4:
        print("Verdict: The faces are SOMEWHAT similar.")
    else:
        print("Verdict: The faces are LIKELY different people.")


if __name__ == "__main__":
    main() 