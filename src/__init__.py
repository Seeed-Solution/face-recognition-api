"""
FaceEmbed API - 基于Hailo-8的人脸特征提取服务

这是一个高性能的人脸嵌入向量生成API，基于Hailo-8 AI加速器实现。
"""

__version__ = "1.0.0"
__author__ = "FaceEmbed Team"
__email__ = "support@faceembed.ai"

# Import key components to the top level
from .app import app, FaceEmbedService
from utils import HailoAsyncInference

# Configure logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "app",
    "FaceEmbedService", 
    "HailoAsyncInference",
] 