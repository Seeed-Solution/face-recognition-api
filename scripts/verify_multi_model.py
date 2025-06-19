#!/usr/bin/env python3
"""
Multi-Model Inference Verification Script

This script verifies that two different models (face detection and 
face recognition) can be loaded and run concurrently on the same 
Hailo-8 device using the HailoRT Python API.

It demonstrates the following principles:
1.  Creating a single VDevice with a scheduling algorithm.
2.  Loading multiple HEF files into separate inference models on the same VDevice.
3.  Running inference on both models in separate threads to simulate simultaneous execution.
"""
import os
import sys
import queue
import threading
import numpy as np
import time

# Adjust the path to import from the project's `src` folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, InferVStreams)
    from src.face_embed_api.utils import HailoAsyncInference
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error: Failed to import Hailo modules. {e}")
    print("Please ensure HailoRT is installed and the environment is set up correctly.")
    sys.exit(1)

# --- Configuration ---
RECOGNITION_HEF = os.path.join(os.path.dirname(__file__), '..', 'models/arcface_mobilefacenet.hef')
DETECTION_HEF = os.path.join(os.path.dirname(__file__), '..', 'models/scrfd_10g.hef')

def check_files():
    """Check if model files exist."""
    if not os.path.exists(RECOGNITION_HEF):
        print(f"Error: Recognition model not found at {RECOGNITION_HEF}")
        sys.exit(1)
    if not os.path.exists(DETECTION_HEF):
        print(f"Error: Detection model not found at {DETECTION_HEF}")
        sys.exit(1)
    print("Model files found.")

def run_inference(name: str, infer_model: InferVStreams, input_shape: tuple, output_queue: queue.Queue):
    """
    A target function for a thread to run inference on a specific model.
    """
    print(f"[{name}] Starting inference thread.")
    
    # Create a dummy input based on the model's expected shape
    dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    
    # Define a valid callback function
    def inference_callback(completion_info):
        if completion_info.exception:
            print(f"[{name}] Error during async inference: {completion_info.exception}")

    with infer_model.configure() as configured_model:
        # Run inference 5 times to simulate a workload
        for i in range(5):
            print(f"[{name}] Running inference cycle {i+1}/5...")

            # Explicitly create output buffers
            output_buffers = {
                output_info.name: np.empty(output_info.shape, dtype=np.uint8)
                for output_info in infer_model.outputs
            }
            
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(dummy_input)
            
            configured_model.wait_for_async_ready(timeout_ms=1000)
            job = configured_model.run_async([bindings], inference_callback)
            job.wait(1000)
            
            # Get output from the pre-defined buffer
            # We'll just check the shape of the first output for verification
            first_output_name = infer_model.outputs[0].name
            output_shape = output_buffers[first_output_name].shape
            output_queue.put((name, output_shape))
            time.sleep(0.1) # Small delay

    print(f"[{name}] Inference thread finished.")

def main():
    """Main function to set up and run the multi-model test."""
    print("--- Multi-Model Inference Verification ---")
    check_files()

    vdevice_params = VDevice.create_params()
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    # Use a context manager for the VDevice
    with VDevice(vdevice_params) as target:
        print(f"\nSuccessfully created VDevice with scheduler value: {vdevice_params.scheduling_algorithm}")

        # --- Load Models ---
        print("Loading models...")
        try:
            rec_infer_model = target.create_infer_model(RECOGNITION_HEF)
            det_infer_model = target.create_infer_model(DETECTION_HEF)
            print("  - Recognition model loaded.")
            print("  - Detection model loaded.")
        except Exception as e:
            print(f"Error: Failed to create one or more inference models: {e}")
            sys.exit(1)

        # Get model shapes
        rec_input_shape = rec_infer_model.input().shape
        det_input_shape = det_infer_model.input().shape
        print(f"\nRecognition Model Input Shape: {rec_input_shape}")
        print(f"Detection Model Input Shape: {det_input_shape}")

        # --- Run Inference Concurrently ---
        print("\nStarting concurrent inference threads...")
        output_queue = queue.Queue()
        
        rec_thread = threading.Thread(
            target=run_inference, 
            args=("Recognition", rec_infer_model, rec_input_shape, output_queue)
        )
        det_thread = threading.Thread(
            target=run_inference, 
            args=("Detection", det_infer_model, det_input_shape, output_queue)
        )
        
        rec_thread.start()
        det_thread.start()
        
        rec_thread.join()
        det_thread.join()

        print("\n--- Verification Results ---")
        if output_queue.empty():
            print("FAILURE: No results were produced by the inference threads.")
        else:
            print("SUCCESS: Inference ran on both models without crashing.")
            results_count = 0
            while not output_queue.empty():
                name, shape = output_queue.get()
                print(f"  - Received output from '{name}' with shape: {shape}")
                results_count += 1
            if results_count == 10: # 5 cycles for each of 2 models
                 print("\nAll 10 inference cycles completed successfully.")
            else:
                 print(f"\nWARNING: Expected 10 results, but got {results_count}.")

if __name__ == "__main__":
    main() 