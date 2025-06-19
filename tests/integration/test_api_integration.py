#!/usr/bin/env python3
"""
FaceEmbed API集成测试
测试API的实际HTTP接口
"""

import base64
import pytest
import requests
import cv2
import numpy as np
import time
import sys
import os
from unittest.mock import MagicMock
import random

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Mock Hailo platform before importing app to allow running without hardware
sys.modules['hailo_platform'] = MagicMock()
# Mock dependent modules for integration test environment
sys.modules['database'] = MagicMock()
sys.modules['utils'] = MagicMock()

from src.app import app


class TestFaceEmbedAPIIntegration:
    """FaceEmbed API集成测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.api_url = "http://localhost:8000"
        cls.timeout = 20 # Increased timeout for safety
        
        # This setup assumes the server is started externally
        # It's better to wait and check for health
        print("--- Waiting for API server to be ready ---")
        max_retries = 20
        retry_interval = 2
        for i in range(max_retries):
            try:
                response = requests.get(f"{cls.api_url}/health", timeout=cls.timeout)
                if response.status_code == 200:
                    print("✅ API server is ready.")
                    return
            except requests.ConnectionError:
                print(f"🔌 API server not ready yet. Retrying in {retry_interval}s... ({i+1}/{max_retries})")
                time.sleep(retry_interval)
        
        pytest.fail("❌ API server did not start in time. Please start it manually.")
        
    def _create_test_image(self, width=300, height=300):
        """创建测试图像"""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (150, 170), (128, 128, 128), -1)
        return image
    
    def _image_to_base64(self, image):
        """将图像转换为base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def test_api_health_check(self):
        """测试API健康检查"""
        response = requests.get(f"{self.api_url}/health", timeout=self.timeout)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "uptime_ms" in data
        assert "loaded_models" in data
        assert isinstance(data["uptime_ms"], int)
        assert data["uptime_ms"] >= 0
        assert isinstance(data["loaded_models"], list)
        assert "scrfd_10g.hef" in data["loaded_models"]
        assert "arcface_mobilefacenet.hef" in data["loaded_models"]
    
    def test_api_root_endpoint(self):
        """测试API根端点"""
        response = requests.get(f"{self.api_url}/", timeout=self.timeout)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "FaceEmbed API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
    
    def test_api_embed_valid_request(self):
        """测试有效的嵌入请求"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_base64": image_base64,
            "bbox": {"x": 50, "y": 50, "w": 100, "h": 120}
        }
        
        response = requests.post(
            f"{self.api_url}/embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 验证响应结构
        assert "vector" in data
        assert "processing_time_ms" in data
        assert "confidence" in data
        
        # 验证向量
        vector = data["vector"]
        assert len(vector) == 512
        
        # 验证向量归一化
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01
        
        # 验证其他字段
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] > 0
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_api_embed_with_landmarks(self):
        """测试使用关键点进行嵌入"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        # 提供了关键点时，bbox 仍然是必需的，但服务会优先使用关键点进行对齐
        request_data = {
            "image_base64": image_base64,
            "bbox": {"x": 50, "y": 50, "w": 100, "h": 120},
            "landmarks": [
                {"x": 84.0, "y": 90.0},
                {"x": 130.0, "y": 89.0},
                {"x": 107.0, "y": 117.0},
                {"x": 89.0, "y": 142.0},
                {"x": 126.0, "y": 142.0}
            ]
        }
        
        response = requests.post(
            f"{self.api_url}/embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "vector" in data
        vector = data["vector"]
        assert len(vector) == 512
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 0.01

    def test_api_embed_invalid_bbox(self):
        """测试无效边界框"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_base64": image_base64,
            "bbox": {"x": 350, "y": 350, "w": 100, "h": 100}  # 超出边界
        }
        
        response = requests.post(
            f"{self.api_url}/embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 400
    
    def test_api_embed_invalid_image(self):
        """测试无效图像数据"""
        request_data = {
            "image_base64": "invalid_base64_data",
            "bbox": {"x": 50, "y": 50, "w": 100, "h": 100}
        }
        
        response = requests.post(
            f"{self.api_url}/embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 400
    
    def test_api_batch_embed(self):
        """测试批量嵌入请求"""
        images = []
        for i in range(3):
            test_image = self._create_test_image()
            image_base64 = self._image_to_base64(test_image)
            images.append({
                "image_base64": image_base64,
                "bbox": {"x": 50, "y": 50, "w": 100, "h": 120}
            })
        
        request_data = {"images": images}
        
        response = requests.post(
            f"{self.api_url}/batch_embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # 验证响应结构
        assert "vectors" in data
        assert "processing_times" in data
        
        # 验证批量结果
        assert len(data["vectors"]) == 3
        assert len(data["processing_times"]) == 3
        
        # 验证每个向量
        for vector in data["vectors"]:
            assert len(vector) == 512
            norm = np.linalg.norm(vector)
            assert abs(norm - 1.0) < 0.01
    
    def test_api_batch_embed_too_many_images(self):
        """测试批量请求超过限制"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        images = []
        for i in range(11):  # 超过最大10张限制
            images.append({
                "image_base64": image_base64,
                "bbox": {"x": 50, "y": 50, "w": 100, "h": 120}
            })
        
        request_data = {"images": images}
        
        response = requests.post(
            f"{self.api_url}/batch_embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 400
    
    def test_api_embedding_consistency(self):
        """测试嵌入结果的一致性"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_base64": image_base64,
            "bbox": {"x": 50, "y": 50, "w": 100, "h": 120}
        }
        
        # 多次请求获取向量
        vectors = []
        for _ in range(3):
            response = requests.post(f"{self.api_url}/embed", json=request_data, timeout=self.timeout)
            assert response.status_code == 200
            vectors.append(np.array(response.json()["vector"]))
            
        # 比较所有向量是否一致
        for i in range(1, len(vectors)):
            np.testing.assert_allclose(vectors[0], vectors[i], rtol=1e-5, atol=1e-5)

    def test_api_detect(self):
        """测试人脸检测接口"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {"image_base64": image_base64}
        
        response = requests.post(
            f"{self.api_url}/detect", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "faces" in data
        assert "processing_time_ms" in data
        assert "image_width" in data
        assert "image_height" in data
        assert data["image_width"] == 300
        assert data["image_height"] == 300
        
        # 因为我们用的是一个简单的矩形，不确定模型是否能检测到
        # 所以只检查结构
        assert isinstance(data["faces"], list)

    def test_api_detect_and_embed(self):
        """测试检测和嵌入一体化接口"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {"image_base64": image_base64}
        
        response = requests.post(
            f"{self.api_url}/detect_and_embed", 
            json=request_data, 
            timeout=self.timeout
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # 假设至少检测到一个"人脸"
        if data:
            face = data[0]
            assert "bbox" in face
            assert "landmarks" in face
            assert "detection_confidence" in face
            assert "embedding" in face
            
            embedding = face["embedding"]
            assert "vector" in embedding
            assert len(embedding["vector"]) == 512
            norm = np.linalg.norm(embedding["vector"])
            assert abs(norm - 1.0) < 0.01
    
    def test_api_performance(self):
        """测试API性能"""
        test_image = self._create_test_image()
        image_base64 = self._image_to_base64(test_image)
        
        request_data = {
            "image_base64": image_base64,
            "bbox": {"x": 50, "y": 50, "w": 100, "h": 120}
        }
        
        # 预热
        requests.post(f"{self.api_url}/embed", json=request_data, timeout=self.timeout)
        
        # 测量时间
        start_time = time.time()
        num_requests = 5
        for _ in range(num_requests):
            requests.post(f"{self.api_url}/embed", json=request_data, timeout=self.timeout)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_requests
        print(f"\nAverage embedding time over {num_requests} requests: {avg_time:.4f}s")
        assert avg_time < 3 # 期望单次请求在3秒内完成


class TestVectorDBIntegration:
    """Vector Database API集成测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        cls.api_url = "http://localhost:8000"
        cls.timeout = 10
        cls.collection_name = f"test_collection_{random.randint(1000, 9999)}"
        cls.user_id = "test_user_123"

    def _generate_random_vector(self):
        """生成一个随机的512维归一化向量"""
        vec = np.random.rand(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return vec.tolist()

    def test_db_add_search_delete_flow(self):
        """测试完整的 'add -> search -> delete' 流程"""
        # 1. 添加向量
        vector_to_add = self._generate_random_vector()
        add_payload = {
            "collection": self.collection_name,
            "user_id": self.user_id,
            "vector": vector_to_add
        }
        add_response = requests.post(f"{self.api_url}/vectors/add", json=add_payload, timeout=self.timeout)
        assert add_response.status_code == 200
        add_data = add_response.json()
        assert add_data["status"] == "success"
        assert "id" in add_data

        # 2. 搜索该向量，应该能找到
        search_payload = {
            "collection": self.collection_name,
            "vector": vector_to_add,
            "threshold": 0.9,
            "top_k": 1
        }
        search_response = requests.post(f"{self.api_url}/vectors/search", json=search_payload, timeout=self.timeout)
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["status"] == "found"
        assert len(search_data["results"]) == 1
        result = search_data["results"][0]
        assert result["user_id"] == self.user_id
        assert result["similarity"] > 0.99

        # 3. 删除该用户的向量
        delete_payload = {
            "collection": self.collection_name,
            "user_id": self.user_id
        }
        delete_response = requests.post(f"{self.api_url}/vectors/delete", json=delete_payload, timeout=self.timeout)
        assert delete_response.status_code == 200
        delete_data = delete_response.json()
        assert delete_data["status"] == "success"
        assert delete_data["deleted_count"] > 0

        # 4. 再次搜索，应该找不到了
        search_again_response = requests.post(f"{self.api_url}/vectors/search", json=search_payload, timeout=self.timeout)
        assert search_again_response.status_code == 200
        search_again_data = search_again_response.json()
        assert search_again_data["status"] == "not_found"

    def test_db_search_not_found(self):
        """测试搜索一个不存在的向量"""
        # 确保集合是空的
        delete_payload = {"collection": self.collection_name, "user_id": "some_user"}
        requests.post(f"{self.api_url}/vectors/delete", json=delete_payload, timeout=self.timeout)

        # 搜索
        search_payload = {
            "collection": self.collection_name,
            "vector": self._generate_random_vector(),
        }
        response = requests.post(f"{self.api_url}/vectors/search", json=search_payload, timeout=self.timeout)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert len(data["results"]) == 0

    def test_db_add_invalid_vector_dim(self):
        """测试添加一个维度不正确的向量"""
        payload = {
            "collection": self.collection_name,
            "user_id": "invalid_dim_user",
            "vector": [0.1, 0.2, 0.3]  # 维度错误
        }
        response = requests.post(f"{self.api_url}/vectors/add", json=payload, timeout=self.timeout)
        assert response.status_code == 400
        
    def test_db_delete_nonexistent_user(self):
        """测试删除一个不存在的用户向量"""
        payload = {
            "collection": self.collection_name,
            "user_id": "non_existent_user_12345"
        }
        response = requests.post(f"{self.api_url}/vectors/delete", json=payload, timeout=self.timeout)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_found"
        assert data["deleted_count"] == 0 