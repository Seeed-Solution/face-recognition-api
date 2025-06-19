#!/usr/bin/env python3
"""
FaceEmbed API - 基于Hailo-8的人脸特征提取服务
支持单张和批量人脸嵌入向量生成
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import queue
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from skimage.transform import SimilarityTransform

import database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hailo imports
try:
    from hailo_platform import VDevice, HailoSchedulingAlgorithm
    from utils import HailoAsyncInference
except (ImportError, ModuleNotFoundError):
    # This will be raised by HailoAsyncInference, but we add a fallback here
    raise ImportError("HailoRT is not installed or not in the PYTHONPATH.")

# Models
class BBoxModel(BaseModel):
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate") 
    w: int = Field(..., description="Width")
    h: int = Field(..., description="Height")

class LandmarkPoint(BaseModel):
    x: float = Field(..., description="X coordinate of the landmark")
    y: float = Field(..., description="Y coordinate of the landmark")

class DetectedFace(BaseModel):
    bbox: BBoxModel = Field(..., description="Face bounding box")
    landmarks: List[LandmarkPoint] = Field(..., description="5 face landmarks")
    confidence: float = Field(..., description="Detection confidence score")

class EmbedRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG image")
    bbox: BBoxModel = Field(..., description="Face bounding box")
    landmarks: Optional[List[LandmarkPoint]] = Field(None, description="List of 5 face landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)")

class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG image")
    confidence_threshold: Optional[float] = Field(0.55, description="Minimum detection confidence threshold. Recommended: 0.55")
    nms_threshold: Optional[float] = Field(0.45, description="Non-Maximum Suppression (NMS) threshold. Recommended: 0.45")
    min_face_size: Optional[int] = Field(8, description="Minimum face size in pixels (width or height) to be considered a valid detection.")

class BatchEmbedRequest(BaseModel):
    images: List[EmbedRequest] = Field(..., description="List of images with bboxes and optional landmarks")

class EmbedResponse(BaseModel):
    vector: List[float] = Field(..., description="512-D face embedding vector")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    confidence: float = Field(..., description="Face quality confidence score")

class DetectResponse(BaseModel):
    faces: List[DetectedFace] = Field(..., description="List of detected faces")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")

class BatchEmbedResponse(BaseModel):
    vectors: List[List[float]] = Field(..., description="List of 512-D face embedding vectors")
    processing_times: List[int] = Field(..., description="Processing times in milliseconds")

# --- New Models for Vector DB ---
class AddVectorRequest(BaseModel):
    collection: str = Field(..., description="Collection name to add the vector to")
    user_id: str = Field(..., description="User ID associated with the vector")
    vector: List[float] = Field(..., description="512-D face embedding vector")

class AddVectorResponse(BaseModel):
    status: str = "success"
    id: int = Field(..., description="The unique ID of the stored vector")
    message: str

class SearchVectorRequest(BaseModel):
    collection: str = Field(..., description="Collection name to search in")
    vector: List[float] = Field(..., description="Query vector for similarity search")
    threshold: Optional[float] = Field(0.32, description="Similarity threshold")
    top_k: Optional[int] = Field(1, description="Number of top results to return")

class SearchResultItem(BaseModel):
    user_id: str
    similarity: float
    id: int

class SearchVectorResponse(BaseModel):
    status: str
    results: List[SearchResultItem]

class DeleteVectorRequest(BaseModel):
    collection: str = Field(..., description="Collection to delete from")
    user_id: str = Field(..., description="The user_id of the vectors to delete")

class DeleteVectorResponse(BaseModel):
    status: str
    deleted_count: int

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str = Field(..., description="Service status")
    uptime_ms: int = Field(..., description="Service uptime in milliseconds")
    loaded_models: List[str] = Field(..., description="List of currently loaded model hef files.")

class DetectAndEmbedResponseItem(BaseModel):
    bbox: BBoxModel
    landmarks: List[LandmarkPoint]
    detection_confidence: float
    embedding: EmbedResponse

# Face Embedding Service
class FaceEmbedService:
    def __init__(self):
        self.start_time = time.time()

        # Debug image saving settings from environment variables
        self.debug_save_images = os.getenv('DEBUG_SAVE_IMAGES', 'false').lower() in ('true', '1', 't')
        self.debug_save_interval_s = int(os.getenv('DEBUG_SAVE_INTERVAL_S', '10'))
        self.last_save_time = 0
        self.debug_image_dir = "debug_images"
        if self.debug_save_images:
            os.makedirs(self.debug_image_dir, exist_ok=True)
            logger.info(f"Debug image saving is enabled. Images will be saved to '{self.debug_image_dir}'.")
        
        # Model paths
        self.face_recognition_hef = os.getenv(
            'FACE_RECOGNITION_HEF', 
            os.path.join(os.path.dirname(__file__), '..', 'models', 'arcface_mobilefacenet.hef')
        )
        self.face_detection_hef = os.getenv(
            'FACE_DETECTION_HEF',
            os.path.join(os.path.dirname(__file__), '..', 'models', 'scrfd_10g.hef')
        )
        
        # Pre-check that model files exist to fail early
        if not os.path.exists(self.face_recognition_hef):
            raise FileNotFoundError(f"Recognition model not found: {self.face_recognition_hef}")
        if not os.path.exists(self.face_detection_hef):
            raise FileNotFoundError(f"Detection model not found: {self.face_detection_hef}")

        # --- Multi-model Hailo setup ---
        self.target = None
        self.det_infer_model = None
        self.rec_infer_model = None
        self.det_input_queue = queue.Queue(maxsize=20)
        self.det_output_queue = queue.Queue(maxsize=20)
        self.rec_input_queue = queue.Queue(maxsize=20)
        self.rec_output_queue = queue.Queue(maxsize=20)
        self.det_quant_infos = {} # To store dequantization parameters
        self.rec_quant_infos = {} # To store dequantization parameters for recognition model
        self.det_thread = None
        self.rec_thread = None
        self._initialize_hailo_device()

    def _initialize_hailo_device(self):
        """Initializes a single VDevice and loads all models onto it."""
        logger.info("Initializing Hailo VDevice and loading models...")
        try:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self.target = VDevice(params)

            logger.info(f"Loading detection model: {self.face_detection_hef}")
            self.det_infer_model = self.target.create_infer_model(self.face_detection_hef)
            
            # --- Extract and store quantization parameters for the detection model ---
            vstream_infos = self.det_infer_model.hef.get_output_vstream_infos()
            self.det_quant_infos = {
                info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
                for info in vstream_infos
            }
            logger.info("--- Detection Model Quantization Info ---")
            for name, params in self.det_quant_infos.items():
                logger.info(f"Layer: {name}, Scale: {params[0]:.4f}, Zero-Point: {params[1]}")
            logger.info("-------------------------------------------")

            logger.info(f"Loading recognition model: {self.face_recognition_hef}")
            self.rec_infer_model = self.target.create_infer_model(self.face_recognition_hef)

            # --- Extract and store quantization parameters for the recognition model ---
            rec_vstream_infos = self.rec_infer_model.hef.get_output_vstream_infos()
            self.rec_quant_infos = {
                info.name: (info.quant_info.qp_scale, info.quant_info.qp_zp)
                for info in rec_vstream_infos
            }
            logger.info("--- Recognition Model Quantization Info ---")
            for name, params in self.rec_quant_infos.items():
                logger.info(f"Layer: {name}, Scale: {params[0]:.4f}, Zero-Point: {params[1]}")
            logger.info("-------------------------------------------")

            # Start inference threads for each model
            self.det_thread = threading.Thread(
                target=self._run_inference_loop, 
                args=("detection", self.det_infer_model, self.det_input_queue, self.det_output_queue),
                daemon=True
            )
            self.rec_thread = threading.Thread(
                target=self._run_inference_loop,
                args=("recognition", self.rec_infer_model, self.rec_input_queue, self.rec_output_queue),
                daemon=True
            )
            self.det_thread.start()
            self.rec_thread.start()

            logger.info("All models loaded and inference threads started.")
        except Exception as e:
            logger.error(f"Failed to initialize Hailo device or models: {e}", exc_info=True)
            raise RuntimeError("Hailo initialization failed.") from e

    def _run_inference_loop(self, name: str, infer_model, input_queue: queue.Queue, output_queue: queue.Queue):
        """
        Generic inference loop for a given model.
        This function runs in a dedicated thread for each model.
        """
        def inference_callback(completion_info):
            if completion_info.exception:
                logger.error(f"[{name}] Async inference error: {completion_info.exception}")

        with infer_model.configure() as configured_model:
            while True:
                batch_data = input_queue.get()
                if batch_data is None:
                    logger.info(f"[{name}] Stopping inference loop.")
                    break
                
                original_frame, preprocessed_frame = batch_data

                try:
                    # Explicitly create output buffers with the correct dtype
                    output_buffers = {
                        info.name: np.empty(info.shape, dtype=np.uint8)
                        for info in infer_model.outputs
                    }
                    bindings = configured_model.create_bindings(output_buffers=output_buffers)
                    bindings.input().set_buffer(preprocessed_frame)
                    
                    configured_model.wait_for_async_ready(timeout_ms=10000)
                    job = configured_model.run_async([bindings], inference_callback)
                    job.wait(10000)

                    # For multi-output models, return a dict. For single, the raw array.
                    if len(output_buffers) == 1:
                        result = list(output_buffers.values())[0]
                    else:
                        result = output_buffers
                    
                    output_queue.put((original_frame, result))

                except Exception as e:
                    logger.error(f"[{name}] Inference execution failed: {e}", exc_info=True)
                    # Propagate exception to the caller
                    output_queue.put((original_frame, e))

    def _decode_image(self, image_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        try:
            image_data = base64.b64decode(image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image")
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")
    
    def _crop_face(self, image: np.ndarray, bbox: BBoxModel) -> Tuple[np.ndarray, float]:
        """Crop face from image using bounding box"""
        h, w = image.shape[:2]
        
        # Validate bbox
        if bbox.x < 0 or bbox.y < 0 or bbox.x + bbox.w > w or bbox.y + bbox.h > h:
            raise ValueError(f"Bounding box is out of image bounds. Image size: ({w}x{h}), BBox: ({bbox.x}, {bbox.y}, {bbox.w}, {bbox.h})")
        
        if bbox.w <= 0 or bbox.h <= 0:
            raise ValueError("Invalid bounding box dimensions")
        
        # Crop face region
        face_image = image[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
        
        # Calculate confidence based on face size and aspect ratio
        face_area = bbox.w * bbox.h
        total_area = w * h
        area_ratio = face_area / total_area
        
        # Aspect ratio score (ideal face aspect ratio ~0.75)
        aspect_ratio = bbox.h / bbox.w
        aspect_score = 1.0 - abs(aspect_ratio - 0.75) / 0.75
        
        # Size score (prefer larger faces)
        size_score = min(area_ratio * 10, 1.0)
        
        confidence = (aspect_score + size_score) / 2.0
        confidence = max(0.1, min(1.0, confidence))
        
        return face_image, confidence
    
    def _align_face(self, image: np.ndarray, landmarks: List[LandmarkPoint]) -> np.ndarray:
        """
        Aligns a face using a similarity transformation based on landmarks.
        This implementation is a Python version of the logic in the provided
        face_align.cpp, using scikit-image for the transformation calculation.
        """
        # Destination landmarks based on the C++ reference code (standard for ArcFace)
        # Scaled for a 112x112 image.
        DEST_LANDMARKS = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        # Convert source landmarks from Pydantic model to numpy array
        src_landmarks = np.array([[p.x, p.y] for p in landmarks], dtype=np.float32)
        
        # Estimate the transformation matrix
        tform = SimilarityTransform()
        tform.estimate(src_landmarks, DEST_LANDMARKS)
        M = tform.params[0:2, :]

        # Get model input shape to determine the output size
        input_shape = self.rec_infer_model.input().shape
        model_h, model_w = int(input_shape[0]), int(input_shape[1])

        # Apply the warp
        aligned_face = cv2.warpAffine(image, M, (model_w, model_h), borderValue=0.0)

        return aligned_face
    
    def _preprocess_face_for_hailo(self, face_image: np.ndarray, model_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Preprocesses face image for the Hailo model, maintaining aspect ratio.
        Resizes the image to fit within the model's input dimensions and pads
        the remaining area with black.
        """
        # Get model input dimensions
        model_h, model_w, _ = model_shape

        # Get original image dimensions
        h, w = face_image.shape[:2]
 
        if h == 0 or w == 0:
            raise ValueError("Input image for preprocessing has zero height or width")

        # Calculate scaling factor to maintain aspect ratio
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized_face = cv2.resize(face_image, (new_w, new_h))

        # Create a black canvas of model input size
        if len(face_image.shape) == 3:
            padded_image = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        else:
            # Handle grayscale images if necessary
            padded_image = np.zeros((model_h, model_w), dtype=np.uint8)

        # Calculate padding to center the image
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2

        # Paste the resized image onto the center of the black canvas
        padded_image[top:top + new_h, left:left + new_w] = resized_face
        
        return padded_image
    
    def _extract_embedding_hailo(self, face_image: np.ndarray, confidence: float, is_aligned: bool = False) -> np.ndarray:
        """Extract face embedding using Hailo model"""
        if not self.rec_infer_model:
            raise RuntimeError("Hailo recognition model not initialized")
        
        # 预处理人脸图像
        model_shape = self.rec_infer_model.input().shape
        preprocessed_face = self._preprocess_face_for_hailo(face_image, model_shape)
        
        # DEBUG: Conditionally save preprocessed image for embedding
        if self.debug_save_images:
            current_time = time.time()
            if (current_time - self.last_save_time) > self.debug_save_interval_s:
                try:
                    prefix = "aligned" if is_aligned else "cropped"
                    filename = os.path.join(
                        self.debug_image_dir, 
                        f"{prefix}_for_embedding_{int(current_time)}_{confidence:.2f}.jpg"
                    )
                    cv2.imwrite(filename, preprocessed_face)
                    logger.info(f"Saved debug image for embedding to {filename}")
                    self.last_save_time = current_time
                except Exception as e:
                    logger.error(f"Failed to save embedding debug image: {e}")
        
        # 发送到推理队列
        self.rec_input_queue.put((face_image, preprocessed_face))
        
        # 获取推理结果
        try:
            original_frame, result = self.rec_output_queue.get(timeout=15.0)
            if isinstance(result, Exception):
                raise RuntimeError("Inference failed in worker thread.") from result
            
            # The recognition model has one output.
            # Handle both dict and raw array for compatibility with old inference loop
            if isinstance(result, dict):
                output_name = list(result.keys())[0]
                embedding_raw = result[output_name]
            else:
                output_name = self.rec_infer_model.hef.get_output_vstream_infos()[0].name
                embedding_raw = result
            
            # --- CRITICAL FIX: Dequantize the output ---
            if output_name in self.rec_quant_infos:
                quant_scale, quant_zp = self.rec_quant_infos[output_name]
                embedding_dequantized = (embedding_raw.astype(np.float32) - quant_zp) * quant_scale
            else:
                logger.warning(f"Quantization info for output '{output_name}' not found. Using raw output.")
                embedding_dequantized = embedding_raw.astype(np.float32)

            # 展平到1维
            embedding = embedding_dequantized.flatten()
            
            # 如果不是512维，进行填充或截断
            if len(embedding) != 512:
                logger.warning(f"Embedding dimension is {len(embedding)}, not 512. Padding/truncating.")
                if len(embedding) > 512:
                    embedding = embedding[:512]
                else:
                    # 填充到512维
                    padding = np.zeros(512 - len(embedding))
                    embedding = np.concatenate([embedding, padding])
            
            # L2归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.astype(np.float32)
            
        except queue.Empty:
            raise RuntimeError("Hailo inference timeout after 15 seconds")
    
    async def extract_embedding(self, request: EmbedRequest) -> Tuple[List[float], int, float]:
        """Extract face embedding from image"""
        start_time = time.time()
        
        try:
            # Decode image
            image = self._decode_image(request.image_base64)
            is_aligned = False

            # If landmarks are provided, align the face first on the whole image
            if request.landmarks and len(request.landmarks) == 5:
                aligned_face = self._align_face(image, request.landmarks)
                # After alignment, the face is already cropped and sized for the model.
                # We can use a dummy confidence score.
                face_image = aligned_face
                confidence = 1.0 
                is_aligned = True
                logger.info("Performed face alignment using landmarks.")
            else:
                # Fallback to original crop and resize logic if no landmarks
                face_image, confidence = self._crop_face(image, request.bbox)
                logger.info("No landmarks provided, using simple crop and resize.")

            # Extract embedding using Hailo hardware
            embedding = self._extract_embedding_hailo(face_image, confidence, is_aligned=is_aligned)
            logger.info("Used Hailo hardware for inference")
            
            # Convert to list
            vector = embedding.tolist()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return vector, processing_time, confidence
            
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    async def extract_embeddings_batch(self, requests: List[EmbedRequest]) -> Tuple[List[List[float]], List[int]]:
        """Extract face embeddings for multiple images"""
        vectors = []
        processing_times = []
        
        for req in requests:
            vector, proc_time, _ = await self.extract_embedding(req)
            vectors.append(vector)
            processing_times.append(proc_time)
        
        return vectors, processing_times

    def _preprocess_image_for_detection(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for face detection model"""
        if not self.det_infer_model:
            raise RuntimeError("Face detection model not initialized")
        
        # Get detection model input dimensions
        input_shape = self.det_infer_model.input().shape
        model_h, model_w = int(input_shape[0]), int(input_shape[1])

        # Get original image dimensions
        h, w = image.shape[:2]

        # Calculate scaling factor to maintain aspect ratio
        scale = min(model_w / w, model_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h))

        # Create a black canvas of model input size
        if len(image.shape) == 3:
            padded_image = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        else:
            padded_image = np.zeros((model_h, model_w), dtype=np.uint8)

        # Calculate padding to center the image
        top = (model_h - new_h) // 2
        left = (model_w - new_w) // 2

        # Paste the resized image onto the center of the black canvas
        padded_image[top:top + new_h, left:left + new_w] = resized_image
        
        return padded_image, scale, (left, top)

    def _non_maximum_suppression(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Performs Non-Maximum Suppression (NMS) on bounding boxes.
        
        Args:
            boxes (np.ndarray): Bounding boxes, shape (N, 4) with format [x1, y1, x2, y2].
            scores (np.ndarray): Confidence scores for each box, shape (N,).
            iou_threshold (float): IoU threshold for suppression.
            
        Returns:
            List[int]: Indices of the boxes to keep.
        """
        if boxes.shape[0] == 0:
            return []

        # Sort by score in descending order
        idxs = scores.argsort()[::-1]

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1) * (y2 - y1)
        
        keep = []
        while idxs.size > 0:
            # Pick the box with the highest score
            i = idxs[0]
            keep.append(i)
            
            # Compute IoU of the picked box with the rest
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            
            intersection = w * h
            union = area[i] + area[idxs[1:]] - intersection
            
            iou = intersection / union
            
            # Keep only boxes with IoU less than the threshold
            remaining_idxs = np.where(iou <= iou_threshold)[0]
            idxs = idxs[remaining_idxs + 1]
            
        return keep

    def _generate_anchors(self, model_input_shape: Tuple[int, int], strides: List[int] = [8, 16, 32], num_anchors: int = 2) -> Dict[int, np.ndarray]:
        """Generate anchors for SCRFD model."""
        all_anchors = {}
        for stride in strides:
            feature_map_h = model_input_shape[0] // stride
            feature_map_w = model_input_shape[1] // stride
            
            # Create a grid of anchor centers
            x_centers = (np.arange(feature_map_w) + 0.5) * stride
            y_centers = (np.arange(feature_map_h) + 0.5) * stride
            
            # Use meshgrid to create all combinations of x and y
            xv, yv = np.meshgrid(x_centers, y_centers)
            
            # Stack and reshape to get a list of (x, y) centers
            anchor_centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
            
            # SCRFD uses num_anchors (typically 2) anchors of the same size at each location
            # So we repeat the centers for each anchor
            total_anchors_at_stride = anchor_centers.shape[0] * num_anchors
            
            # Repeat each center 'num_anchors' times
            repeated_centers = np.repeat(anchor_centers, num_anchors, axis=0)
            
            # Add stride as a third column for decoding later
            stride_col = np.full((total_anchors_at_stride, 1), stride)
            
            # Final anchor format for this stride: [center_x, center_y, stride]
            anchors_with_stride = np.concatenate([repeated_centers, stride_col], axis=1)
            
            all_anchors[stride] = anchors_with_stride
        return all_anchors

    def _parse_detection_results(self, 
                                 raw_outputs: Dict[str, np.ndarray], 
                                 original_image_shape: Tuple[int, int],
                                 model_input_shape: Tuple[int, int],
                                 scale: float, 
                                 offset: Tuple[int, int], 
                                 confidence_threshold: float,
                                 nms_threshold: float, 
                                 min_face_size: int) -> List[DetectedFace]:
        """
        Parses the raw output from the SCRFD model, decodes bounding boxes
        and landmarks, performs NMS, and scales back to original image size.
        """
        strides = [8, 16, 32]
        num_anchors = 2

        # 1. Generate anchors
        anchors = self._generate_anchors(model_input_shape, strides, num_anchors)

        all_proposals = []

        # 2. Decode outputs for each stride
        for stride in strides:
            # --- Get the raw outputs for this stride ---
            # NOTE: The exact output names depend on the exported model.
            # We derive them based on the pattern found in the logs.
            # Stride 8: conv41(score), conv42(bbox), conv43(kps)
            # Stride 16: conv49(score), conv50(bbox), conv51(kps)
            # Stride 32: conv56(score), conv57(bbox), conv58(kps)
            score_layer_name = f'scrfd_10g/conv{41 + (stride // 16 * 8)}'
            bbox_layer_name = f'scrfd_10g/conv{42 + (stride // 16 * 8)}'
            kps_layer_name = f'scrfd_10g/conv{43 + (stride // 16 * 8)}'
            
            if stride == 32: # Special case for stride 32 names
                score_layer_name = 'scrfd_10g/conv56'
                bbox_layer_name = 'scrfd_10g/conv57'
                kps_layer_name = 'scrfd_10g/conv58'


            scores_raw = raw_outputs.get(score_layer_name)
            bbox_deltas_raw = raw_outputs.get(bbox_layer_name)
            kps_deltas_raw = raw_outputs.get(kps_layer_name)

            if scores_raw is None or bbox_deltas_raw is None or kps_deltas_raw is None:
                logger.warning(f"Missing one or more output layers for stride {stride}. Skipping.")
                continue

            # --- Dequantize and Reshape ---
            score_scale, score_zp = self.det_quant_infos[score_layer_name]
            bbox_scale, bbox_zp = self.det_quant_infos[bbox_layer_name]
            kps_scale, kps_zp = self.det_quant_infos[kps_layer_name]

            scores = (scores_raw.astype(np.float32) - score_zp) * score_scale
            scores = scores.reshape(-1, 1)

            bbox_deltas = (bbox_deltas_raw.astype(np.float32) - bbox_zp) * bbox_scale
            bbox_deltas = bbox_deltas.reshape(-1, 4)

            kps_deltas = (kps_deltas_raw.astype(np.float32) - kps_zp) * kps_scale
            kps_deltas = kps_deltas.reshape(-1, 10) # 5 landmarks * 2 coords
            
            # --- Get Anchors for this stride ---
            current_anchors = anchors[stride]
            
            # --- Filter by confidence threshold ---
            keep_indices = np.where(scores >= confidence_threshold)[0]
            if keep_indices.shape[0] == 0:
                continue

            scores = scores[keep_indices]
            bbox_deltas = bbox_deltas[keep_indices]
            kps_deltas = kps_deltas[keep_indices]
            current_anchors = current_anchors[keep_indices]

            # --- Decode Bounding Boxes ---
            # Formula: new_coord = anchor_center + delta * stride
            anchor_cx = current_anchors[:, 0]
            anchor_cy = current_anchors[:, 1]
            
            # bbox decoding: The model predicts distance to the 4 sides from the anchor center
            x1 = anchor_cx - bbox_deltas[:, 0] * stride
            y1 = anchor_cy - bbox_deltas[:, 1] * stride
            x2 = anchor_cx + bbox_deltas[:, 2] * stride
            y2 = anchor_cy + bbox_deltas[:, 3] * stride
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # --- Decode Landmarks ---
            decoded_kps = np.zeros_like(kps_deltas)
            for i in range(5):
                # kps decoding: The model predicts offset from the anchor center
                kps_x = anchor_cx + kps_deltas[:, i * 2] * stride
                kps_y = anchor_cy + kps_deltas[:, i * 2 + 1] * stride
                decoded_kps[:, i * 2] = kps_x
                decoded_kps[:, i * 2 + 1] = kps_y

            # Combine proposals from this stride
            # Format: [x1, y1, x2, y2, score, kps...]
            proposals = np.concatenate([decoded_boxes, scores, decoded_kps], axis=1)
            all_proposals.append(proposals)

        if not all_proposals:
            return []

        # 3. Combine all proposals and perform NMS
        all_proposals = np.concatenate(all_proposals, axis=0)

        boxes_for_nms = all_proposals[:, :4]
        scores_for_nms = all_proposals[:, 4]

        keep_indices = self._non_maximum_suppression(boxes_for_nms, scores_for_nms, nms_threshold)
        
        final_proposals = all_proposals[keep_indices]

        # 4. Scale back to original image and format the output
        final_faces = []
        h_orig, w_orig = original_image_shape[:2]
        offset_x, offset_y = offset
        
        for proposal in final_proposals:
            # Scale coordinates from padded/resized model input space to original image space
            x1 = (proposal[0] - offset_x) / scale
            y1 = (proposal[1] - offset_y) / scale
            x2 = (proposal[2] - offset_x) / scale
            y2 = (proposal[3] - offset_y) / scale

            # Clamp to image bounds
            x1 = max(0, min(w_orig, x1))
            y1 = max(0, min(h_orig, y1))
            x2 = max(0, min(w_orig, x2))
            y2 = max(0, min(h_orig, y2))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # --- Filter by minimum size ---
            if min(bbox_w, bbox_h) < min_face_size:
                continue

            # Scale landmarks
            landmarks = []
            for i in range(5):
                kpt_x = (proposal[5 + i * 2] - offset_x) / scale
                kpt_y = (proposal[5 + i * 2 + 1] - offset_y) / scale
                landmarks.append(LandmarkPoint(x=float(kpt_x), y=float(kpt_y)))

            face = DetectedFace(
                bbox=BBoxModel(x=int(x1), y=int(y1), w=int(bbox_w), h=int(bbox_h)),
                landmarks=landmarks,
                confidence=float(proposal[4])
            )
            final_faces.append(face)
        
        return final_faces

    async def detect_faces(self, request: DetectRequest) -> Tuple[List[DetectedFace], int, int, int]:
        """INTERNAL: Detect faces in image using Hailo face detection model"""
        start_time = time.time()
        
        try:
            # Decode image
            image = self._decode_image(request.image_base64)
            h, w = image.shape[:2]
            
            # Preprocess image for detection
            input_shape = self.det_infer_model.input().shape
            preprocessed_image, scale, offset = self._preprocess_image_for_detection(image)
            
            # Send to detection inference queue
            self.det_input_queue.put((image, preprocessed_image))
            
            # Get detection results
            try:
                original_frame, results = self.det_output_queue.get(timeout=15.0)
                if isinstance(results, Exception):
                    raise RuntimeError("Detection inference failed in worker thread.") from results

                # The new parsing function expects a dictionary of outputs
                if not isinstance(results, dict):
                    raise TypeError(f"Expected a dict of outputs from inference, but got {type(results)}")

                # Parse detection results using the new SCRFD-specific logic
                faces = self._parse_detection_results(
                    raw_outputs=results,
                    original_image_shape=(h, w),
                    model_input_shape=(input_shape[0], input_shape[1]),
                    scale=scale,
                    offset=offset,
                    confidence_threshold=request.confidence_threshold,
                    nms_threshold=request.nms_threshold,
                    min_face_size=request.min_face_size
                )
                
                # --- Debug: Save image with detected faces ---
                if self.debug_save_images and faces:
                    try:
                        # Draw bounding boxes on the original image
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        for face in faces:
                            bbox = face.bbox
                            p1 = (bbox.x, bbox.y)
                            p2 = (bbox.x + bbox.w, bbox.y + bbox.h)
                            cv2.rectangle(image_bgr, p1, p2, (0, 255, 0), 2)
                            
                            if hasattr(face, "landmarks") and face.landmarks is not None:
                                for landmark in face.landmarks:
                                    cv2.circle(image_bgr, (int(landmark.x), int(landmark.y)), 2, (0, 0, 255), -1)
                        
                        # Save the image
                        timestamp = int(time.time())
                        filename = os.path.join(self.debug_image_dir, f"detected_{timestamp}_{len(faces)}_faces.jpg")
                        cv2.imwrite(filename, image_bgr)
                        logger.info(f"Saved debug image with {len(faces)} detections to {filename}")
                    except Exception as e:
                        logger.error(f"Failed to save debug image: {e}")

                processing_time = int((time.time() - start_time) * 1000)
                
                return faces, processing_time, w, h
                
            except queue.Empty:
                raise RuntimeError("Face detection inference timeout after 15 seconds")
                
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def detect_and_embed(self, request: DetectRequest):
        """Detects faces and returns their embeddings in one go."""
        total_start_time = time.time()
        
        # Step 1: Detect faces
        logger.info("Step 1: Detecting faces...")
        detected_faces, _, _, _ = await self.detect_faces(request)
        
        if not detected_faces:
            logger.info("No faces detected.")
            return []

        logger.info(f"Detected {len(detected_faces)} faces. Step 2: Extracting embeddings...")
        
        # Step 2: Extract embeddings for each detected face
        results = []
        original_image = self._decode_image(request.image_base64)

        for face in detected_faces:
            embed_req = EmbedRequest(
                image_base64=request.image_base64, # Pass the original b64
                bbox=face.bbox,
                landmarks=face.landmarks
            )
            # We call the internal logic, not the full endpoint async method
            try:
                start_time = time.time()
                is_aligned = False
                # Ensure the recognition model is ready for its specific inputs
                if embed_req.landmarks and len(embed_req.landmarks) == 5:
                    aligned_face = self._align_face(original_image, embed_req.landmarks)
                    face_image = aligned_face
                    confidence = 1.0
                    is_aligned = True
                else:
                    face_image, confidence = self._crop_face(original_image, embed_req.bbox)

                embedding = self._extract_embedding_hailo(face_image, confidence, is_aligned=is_aligned)
                vector = embedding.tolist()
                processing_time = int((time.time() - start_time) * 1000)
                
                # Append a dictionary with all info
                results.append({
                    "bbox": face.bbox.model_dump(),
                    "landmarks": [lm.model_dump() for lm in face.landmarks],
                    "detection_confidence": face.confidence,
                    "embedding": {
                        "vector": vector,
                        "processing_time_ms": processing_time,
                        "confidence": confidence
                    }
                })
            except Exception as e:
                logger.error(f"Could not process embedding for a face: {e}")

        total_processing_time = int((time.time() - total_start_time) * 1000)
        logger.info(f"Finished detect_and_embed in {total_processing_time}ms")
        return results

    def get_health(self) -> HealthResponse:
        """Get service health status"""
        uptime_ms = int((time.time() - self.start_time) * 1000)
        
        loaded = []
        if self.det_infer_model:
            loaded.append(os.path.basename(self.face_detection_hef))
        if self.rec_infer_model:
            loaded.append(os.path.basename(self.face_recognition_hef))

        return HealthResponse(
            status="ok",
            uptime_ms=uptime_ms,
            loaded_models=loaded or ["none"]
        )
    
    def __del__(self):
        """Cleanup resources"""
        logger.info("FaceEmbedService shutting down. Unloading model.")
        if self.det_input_queue:
            self.det_input_queue.put(None)
        if self.rec_input_queue:
            self.rec_input_queue.put(None)

        if self.det_thread and self.det_thread.is_alive():
            self.det_thread.join(timeout=5.0)
        if self.rec_thread and self.rec_thread.is_alive():
            self.rec_thread.join(timeout=5.0)
        
        logger.info("Inference threads stopped.")

# Global service instance - will be initialized lazily
face_embed_service = None

def get_face_embed_service():
    """Get or create face embed service instance"""
    global face_embed_service
    if face_embed_service is None:
        face_embed_service = FaceEmbedService()
    return face_embed_service

# FastAPI app with optimized settings for concurrent access
app = FastAPI(
    title="FaceEmbed API",
    description="基于Hailo-8的人脸特征提取服务 - 支持多设备并发访问",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware - optimized for cross-machine access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许Node-RED服务器跨域访问
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    return response

# Application startup event
@app.on_event("startup")
def on_startup():
    """Initialize database on startup."""
    logger.info("Application startup...")
    database.init_db()

# Routes
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return get_face_embed_service().get_health()

@app.post("/embed", response_model=EmbedResponse)
async def embed_face(request: EmbedRequest):
    """Extract face embedding from single image"""
    service = get_face_embed_service()
    vector, processing_time, confidence = await service.extract_embedding(
        request
    )
    
    return EmbedResponse(
        vector=vector,
        processing_time_ms=processing_time,
        confidence=confidence
    )

@app.post("/batch_embed", response_model=BatchEmbedResponse)
async def batch_embed_faces(request: BatchEmbedRequest):
    """Extract face embeddings from multiple images"""
    if len(request.images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    service = get_face_embed_service()
    vectors, processing_times = await service.extract_embeddings_batch(request.images)
    
    return BatchEmbedResponse(
        vectors=vectors,
        processing_times=processing_times
    )

@app.post("/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """Detect faces in image using Hailo face detection model"""
    service = get_face_embed_service()
    faces, processing_time, image_width, image_height = await service.detect_faces(request)
    
    return DetectResponse(
        faces=faces,
        processing_time_ms=processing_time,
        image_width=image_width,
        image_height=image_height
    )

@app.post("/detect_and_embed", response_model=List[DetectAndEmbedResponseItem])
async def detect_and_embed_faces(request: DetectRequest):
    """Detect faces in an image and return their embeddings"""
    service = get_face_embed_service()
    results = await service.detect_and_embed(request)
    return results

# --- Vector Database Endpoints ---

@app.post("/vectors/add", response_model=AddVectorResponse)
async def add_vector_endpoint(request: AddVectorRequest):
    """Adds a face vector to the SQLite database."""
    try:
        vector_np = np.array(request.vector, dtype=np.float32)
        if vector_np.shape != (512,):
            raise ValueError(f"Invalid vector dimensions. Expected 512, got {vector_np.shape[0]}.")
        
        vector_id = database.add_vector(
            collection=request.collection,
            user_id=request.user_id,
            vector=vector_np
        )
        return AddVectorResponse(
            id=vector_id,
            message=f"Vector added to collection '{request.collection}' for user '{request.user_id}'."
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /vectors/add endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while adding vector.")

@app.post("/vectors/search", response_model=SearchVectorResponse)
async def search_vector_endpoint(request: SearchVectorRequest):
    """Searches for similar vectors in the SQLite database."""
    try:
        query_vector = np.array(request.vector, dtype=np.float32)
        if query_vector.shape != (512,):
            raise ValueError(f"Invalid query vector dimensions. Expected 512, got {query_vector.shape[0]}.")

        search_results = database.search_vectors(
            collection=request.collection,
            query_vector=query_vector,
            threshold=request.threshold,
            top_k=request.top_k
        )
        
        # Convert list of dicts to list of SearchResultItem models
        results_models = [SearchResultItem(**item) for item in search_results]

        if not results_models:
            return SearchVectorResponse(status="not_found", results=[])

        return SearchVectorResponse(status="found", results=results_models)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /vectors/search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while searching vectors.")


@app.post("/vectors/delete", response_model=DeleteVectorResponse)
async def delete_vector_endpoint(request: DeleteVectorRequest):
    """Deletes all vectors for a given user_id in a collection."""
    try:
        deleted_count = database.delete_vectors_by_user(
            collection=request.collection,
            user_id=request.user_id
        )
        if deleted_count > 0:
            return DeleteVectorResponse(status="success", deleted_count=deleted_count)
        else:
            return DeleteVectorResponse(status="not_found", deleted_count=0)
    except Exception as e:
        logger.error(f"Error in /vectors/delete endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while deleting vectors.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "FaceEmbed API", "version": "1.0.0", "status": "running"}