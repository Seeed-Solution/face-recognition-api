#!/usr/bin/env python3
"""
FaceEmbed API单元测试
测试人脸嵌入API的各种功能和边界情况
"""

import unittest
import base64
import tempfile
import os
import json
import time
import sys
from typing import List
import asyncio

import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Mock Hailo platform before importing app
sys.modules['hailo_platform'] = MagicMock()

# Mock the database module before importing app
# This ensures that all app-level calls to 'database' are mocked
sys.modules['database'] = MagicMock()
sys.modules['utils'] = MagicMock()

from src.app import (
    app, FaceEmbedService, get_face_embed_service, BBoxModel, DetectRequest,
    DetectedFace, LandmarkPoint, EmbedRequest, HealthResponse,
    AddVectorRequest, SearchVectorRequest, DeleteVectorRequest
)


class TestFaceEmbedAPI(unittest.TestCase):
    """FaceEmbed API端到端测试 - Mocks the entire service layer"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        cls.client = TestClient(app)
    
    def setUp(self):
        """测试准备"""
        self.test_image = self._create_test_image()
        self.test_image_base64 = self._image_to_base64(self.test_image)
        self.valid_bbox = BBoxModel(x=50, y=50, w=100, h=120)
        
    def _create_test_image(self, width=300, height=300):
        """创建测试图像"""
        # 创建白色背景
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 在中心添加一个灰色矩形作为"人脸"
        face_x, face_y = 50, 50
        face_w, face_h = 100, 120
        cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (128, 128, 128), -1)
        
        return image
    
    def _image_to_base64(self, image):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    @patch('src.app.get_face_embed_service')
    def test_health_endpoint(self, mock_get_service):
        """测试健康检查端点"""
        mock_service = Mock(spec=FaceEmbedService)
        mock_service.get_health.return_value = HealthResponse(
            status="ok",
            uptime_ms=1000,
            loaded_models=["mock_model1.hef", "mock_model2.hef"]
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证响应结构
        self.assertIn("status", data)
        self.assertIn("uptime_ms", data)
        self.assertIn("loaded_models", data)
        
        # 验证数据类型
        self.assertIsInstance(data["uptime_ms"], int)
        self.assertTrue(data["uptime_ms"] >= 0)
        self.assertIsInstance(data["loaded_models"], list)
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证基本信息
        self.assertIn("message", data)
        self.assertIn("version", data)
        self.assertIn("status", data)
        self.assertEqual(data["message"], "FaceEmbed API")
    
    @patch('src.app.get_face_embed_service')
    def test_embed_endpoint_valid_request(self, mock_get_service):
        """测试有效的人脸嵌入请求"""
        mock_service = Mock(spec=FaceEmbedService)
        # Mock extract_embedding to return a valid 512-D normalized vector
        mock_vector = np.random.normal(0, 1, 512).astype(np.float32)
        mock_vector = mock_vector / np.linalg.norm(mock_vector)
        
        async def mock_extract_embedding(request):
            return mock_vector.tolist(), 10, 0.8
        mock_service.extract_embedding = mock_extract_embedding
        mock_get_service.return_value = mock_service
        
        request_data = {
            "image_base64": self.test_image_base64,
            "bbox": {
                "x": self.valid_bbox.x,
                "y": self.valid_bbox.y,
                "w": self.valid_bbox.w,
                "h": self.valid_bbox.h
            }
        }
        
        response = self.client.post("/embed", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证响应结构
        self.assertIn("vector", data)
        self.assertIn("processing_time_ms", data)
        self.assertIn("confidence", data)
        
        # 验证向量
        vector = data["vector"]
        self.assertEqual(len(vector), 512)
        
        # 验证向量归一化
        norm = np.linalg.norm(np.array(vector))
        self.assertAlmostEqual(norm, 1.0, places=6)
        
        # 验证置信度范围
        confidence = data["confidence"]
        self.assertTrue(0.0 <= confidence <= 1.0)
        
        # 验证处理时间
        processing_time = data["processing_time_ms"]
        self.assertIsInstance(processing_time, int)
        self.assertTrue(processing_time > 0)
    
    @patch('src.app.get_face_embed_service')
    def test_embed_endpoint_invalid_bbox(self, mock_get_service):
        """测试无效边界框的人脸嵌入"""
        # Mock service to raise ValueError for invalid bbox
        async def mock_extract_embedding_error(request):
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="Bounding box is out of image bounds")
            
        mock_service = Mock(spec=FaceEmbedService)
        mock_service.extract_embedding = mock_extract_embedding_error
        mock_get_service.return_value = mock_service
        
        request_data = {
            "image_base64": self.test_image_base64,
            "bbox": {
                "x": -10,  # 无效坐标
                "y": -10,
                "w": 50,
                "h": 50
            }
        }
        
        response = self.client.post("/embed", json=request_data)
        self.assertEqual(response.status_code, 400)
    
    def test_embed_endpoint_invalid_image(self):
        """测试无效的图像数据"""
        request_data = {
            "image_base64": "invalid_base64_data",
            "bbox": {
                "x": self.valid_bbox.x,
                "y": self.valid_bbox.y,
                "w": self.valid_bbox.w,
                "h": self.valid_bbox.h
            }
        }
        
        response = self.client.post("/embed", json=request_data)
        self.assertEqual(response.status_code, 400)
    
    def test_embed_endpoint_missing_fields(self):
        """测试缺少必要字段的请求"""
        # 缺少bbox字段
        request_data = {
            "image_base64": self.test_image_base64
        }
        
        response = self.client.post("/embed", json=request_data)
        self.assertEqual(response.status_code, 422)  # Validation error
    
    @patch('src.app.get_face_embed_service')
    def test_batch_embed_endpoint(self, mock_get_service):
        """测试批量人脸嵌入"""
        mock_service = Mock(spec=FaceEmbedService)
        mock_vector = np.random.normal(0, 1, 512).astype(np.float32)
        mock_vector = mock_vector / np.linalg.norm(mock_vector)

        async def mock_extract_embeddings_batch(requests):
            vectors = []
            processing_times = []
            for _ in requests:
                vectors.append(mock_vector.tolist())
                processing_times.append(10)
            return vectors, processing_times
        mock_service.extract_embeddings_batch = mock_extract_embeddings_batch
        mock_get_service.return_value = mock_service
        
        # 创建多个测试图像
        images = []
        for i in range(3):
            image = self._create_test_image()
            image_base64 = self._image_to_base64(image)
            images.append({
                "image_base64": image_base64,
                "bbox": {
                    "x": self.valid_bbox.x,
                    "y": self.valid_bbox.y,
                    "w": self.valid_bbox.w,
                    "h": self.valid_bbox.h
                }
            })
        
        request_data = {"images": images}
        
        response = self.client.post("/batch_embed", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # 验证响应结构
        self.assertIn("vectors", data)
        self.assertIn("processing_times", data)
        
        # 验证批量结果
        self.assertEqual(len(data["vectors"]), 3)
        self.assertEqual(len(data["processing_times"]), 3)
        
        # 验证每个向量
        for vector in data["vectors"]:
            self.assertEqual(len(vector), 512)
            norm = np.linalg.norm(np.array(vector))
            self.assertAlmostEqual(norm, 1.0, places=6)
    
    @patch('src.app.get_face_embed_service')
    def test_batch_embed_too_many_images(self, mock_get_service):
        """测试批量嵌入请求数量过多"""
        # The endpoint itself should raise an HTTPException for too many images
        # before the service is even called.
        async def mock_batch_embed(requests):
            # This should not be called
            return [], []

        mock_service = Mock(spec=FaceEmbedService)
        mock_service.extract_embeddings_batch = mock_batch_embed
        mock_get_service.return_value = mock_service
        
        images = [{"image_base64": "test", "bbox": {"x":0,"y":0,"w":1,"h":1}}] * 21
        request_data = {"images": images}
        
        response = self.client.post("/batch_embed", json=request_data)
        self.assertEqual(response.status_code, 400) # As defined in the endpoint logic

    @patch('src.app.get_face_embed_service')
    def test_detect_and_embed_endpoint(self, mock_get_service):
        """测试检测和嵌入端点"""
        mock_service = Mock(spec=FaceEmbedService)
        
        # Mock the combined service function
        async def mock_detect_and_embed(request):
            mock_vector = np.random.rand(512).tolist()
            return [
                {
                    "bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
                    "landmarks": [{"x": 0, "y": 0}] * 5,
                    "detection_confidence": 0.99,
                    "embedding": {
                        "vector": mock_vector,
                        "processing_time_ms": 15,
                        "confidence": 0.9
                    }
                }
            ]
        
        mock_service.detect_and_embed = mock_detect_and_embed
        mock_get_service.return_value = mock_service

        request_data = {"image_base64": self.test_image_base64}
        response = self.client.post("/detect_and_embed", json=request_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIn("embedding", data[0])
        self.assertEqual(len(data[0]["embedding"]["vector"]), 512)

    # --- New tests for Vector DB endpoints ---
    
    @patch('src.app.database')
    def test_add_vector_endpoint_success(self, mock_db_mod):
        """测试成功添加向量的端点"""
        mock_db_mod.add_vector.return_value = 123
        vector = list(np.random.rand(512))
        
        payload = {
            "collection": "test_coll",
            "user_id": "test_user",
            "vector": vector
        }
        
        response = self.client.post("/vectors/add", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["id"], 123)
        mock_db_mod.add_vector.assert_called_once()

    @patch('src.app.database')
    def test_add_vector_invalid_dim(self, mock_db_mod):
        """测试添加无效维度向量"""
        vector = [0.1, 0.2] # Invalid dimension
        payload = {
            "collection": "test_coll",
            "user_id": "test_user",
            "vector": vector
        }
        response = self.client.post("/vectors/add", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("id", response.json())
        mock_db_mod.add_vector.assert_not_called()

    @patch('src.app.database')
    def test_search_vector_endpoint_found(self, mock_db_mod):
        """测试搜索向量并找到结果"""
        mock_results = [{"id": 1, "user_id": "found_user", "similarity": 0.95}]
        mock_db_mod.search_vectors.return_value = mock_results
        
        payload = {
            "collection": "test_coll",
            "vector": list(np.random.rand(512)),
            "threshold": 0.8,
            "top_k": 1
        }
        
        response = self.client.post("/vectors/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "found")
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["user_id"], "found_user")
        mock_db_mod.search_vectors.assert_called_once()

    @patch('src.app.database')
    def test_search_vector_not_found(self, mock_db_mod):
        """测试搜索向量未找到结果"""
        mock_db_mod.search_vectors.return_value = [] # No results
        
        payload = {
            "collection": "test_coll",
            "vector": list(np.random.rand(512))
        }
        
        response = self.client.post("/vectors/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "not_found")
        self.assertEqual(len(data["results"]), 0)

    @patch('src.app.database')
    def test_delete_vector_success(self, mock_db_mod):
        """测试成功删除向量"""
        mock_db_mod.delete_vectors_by_user.return_value = 1 # 1 row deleted
        
        payload = {"collection": "test_coll", "user_id": "user_to_delete"}
        
        response = self.client.post("/vectors/delete", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["deleted_count"], 1)

    @patch('src.app.database')
    def test_delete_vector_not_found(self, mock_db_mod):
        """测试删除不存在的向量"""
        mock_db_mod.delete_vectors_by_user.return_value = 0 # 0 rows deleted
        
        payload = {"collection": "test_coll", "user_id": "non_existent_user"}
        
        response = self.client.post("/vectors/delete", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "not_found")
        self.assertEqual(data["deleted_count"], 0)


class TestFaceEmbedService(unittest.TestCase):
    """FaceEmbedService 单元测试"""
    
    @patch('src.app.FaceEmbedService._initialize_hailo_device')
    def setUp(self, mock_init_hailo):
        """测试准备"""
        # We patch the hailo initialization to avoid hardware dependency
        self.service = FaceEmbedService()
        self.test_image = self._create_test_image()
    
    def tearDown(self):
        """测试清理"""
        # Clean up any created debug files
        if self.service.debug_save_images:
            for f in os.listdir(self.service.debug_image_dir):
                os.remove(os.path.join(self.service.debug_image_dir, f))
        # Manually call cleanup since we are not running a full app lifecycle
        self.service.__del__()

    def _create_test_image(self):
        return np.ones((300, 300, 3), dtype=np.uint8) * 255

    def _image_to_base64(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def test_decode_image_valid(self):
        """测试有效的base64图像解码"""
        image = self._create_test_image()
        b64_str = self._image_to_base64(image)
        decoded_image = self.service._decode_image(b64_str)
        self.assertIsInstance(decoded_image, np.ndarray)
        self.assertEqual(decoded_image.shape, (300, 300, 3))

    def test_decode_image_invalid(self):
        """测试无效的base64图像解码"""
        with self.assertRaises(ValueError):
            self.service._decode_image("invalid-base64")

    def test_crop_face_valid(self):
        """测试有效的人脸裁剪"""
        bbox = BBoxModel(x=50, y=50, w=100, h=120)
        face_img, confidence = self.service._crop_face(self.test_image, bbox)
        self.assertEqual(face_img.shape, (120, 100, 3))
        # Confidence calculation is complex, check if it's within a reasonable range
        self.assertTrue(0.6 <= confidence <= 1.0)

    def test_crop_face_invalid_bbox(self):
        """测试无效边界框的裁剪"""
        bbox = BBoxModel(x=50, y=50, w=0, h=120) # Invalid width
        with self.assertRaises(ValueError):
            self.service._crop_face(self.test_image, bbox)

    def test_crop_face_out_of_bounds(self):
        """测试超出边界的裁剪"""
        # BBox goes beyond the image dimensions (300x300)
        bbox = BBoxModel(x=250, y=250, w=100, h=100)
        with self.assertRaises(ValueError):
            self.service._crop_face(self.test_image, bbox)

    def test_preprocess_face_for_hailo(self):
        """测试Hailo的人脸预处理"""
        # Mock internal components for isolation
        self.service.rec_infer_model = MagicMock()
        # The function takes the target model shape (H, W, C) as input
        model_shape = (112, 112, 3) 
        
        test_face = np.ones((120, 100, 3), dtype=np.uint8)
        
        processed_image = self.service._preprocess_face_for_hailo(test_face, model_shape)
        
        # The function should return an image padded to the model's HxW dimensions
        self.assertEqual(processed_image.shape, (112, 112, 3))

    def test_get_health(self):
        """测试健康状态获取"""
        health = self.service.get_health()
        self.assertEqual(health.status, "ok")
        self.assertGreaterEqual(health.uptime_ms, 0)

    @patch('src.app.queue.Queue')
    @patch('src.app.FaceEmbedService._parse_detection_results')
    def test_detect_faces_logic(self, mock_parse, mock_queue):
        """测试人脸检测逻辑"""
        # Mock Hailo model and queues since they are not available
        self.service.det_infer_model = MagicMock()
        mock_input = MagicMock()
        mock_input.shape = (320, 320, 3)
        self.service.det_infer_model.input.return_value = mock_input
        self.service.det_output_queue = mock_queue.return_value

        # Mock the output from the inference queue
        mock_output = {"mock_layer": np.random.rand(1, 10, 10, 2).astype(np.uint8)}
        mock_queue.return_value.get.return_value = (None, mock_output)
            
        # Run detection
        detect_request = DetectRequest(image_base64=self._image_to_base64(self.test_image))
        faces, _, _, _ = asyncio.run(self.service.detect_faces(detect_request))
            
        # We mocked the parse function, so just check it was called
        mock_parse.assert_called_once()


class TestBBoxModel(unittest.TestCase):
    """BBox Pydantic模型单元测试"""

    def test_valid_bbox(self):
        """测试有效的BBox"""
        bbox = BBoxModel(x=10, y=10, w=100, h=100)
        self.assertEqual(bbox.x, 10)
        self.assertEqual(bbox.w, 100)

    def test_bbox_validation_in_service(self):
        """测试在服务层面的BBox验证 (w/h <= 0)"""
        # Pydantic itself might allow w=0, but the service logic should reject it.
        service = FaceEmbedService()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test case: width is zero
        with self.assertRaisesRegex(ValueError, "Invalid bounding box dimensions"):
            invalid_bbox_w = BBoxModel(x=10, y=10, w=0, h=10)
            service._crop_face(image, invalid_bbox_w)

        # Test case: height is zero
        with self.assertRaisesRegex(ValueError, "Invalid bounding box dimensions"):
            invalid_bbox_h = BBoxModel(x=10, y=10, w=10, h=0)
            service._crop_face(image, invalid_bbox_h)


if __name__ == '__main__':
    unittest.main()
