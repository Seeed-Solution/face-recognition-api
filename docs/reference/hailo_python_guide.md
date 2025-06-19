# Hailo Python API 调用指南

## 概述

本指南基于 [Hailo Application Code Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/hailo-8/python) 整理，提供 Hailo-8 设备的 Python API 使用说明，支持人脸识别、目标检测等 AI 推理任务。

## 核心组件架构

### 1. 主要依赖库

```python
from hailo_platform import (
    HEF, VDevice, FormatType, 
    HailoSchedulingAlgorithm
)
import numpy as np
import cv2
import queue
import threading
```

### 2. 核心类结构

#### HailoAsyncInference 类
- **功能**: 异步推理处理的核心类
- **特点**: 支持批处理、多线程、队列管理
- **适用场景**: 实时视频流处理、大批量图像处理

#### 基本初始化参数
```python
HailoAsyncInference(
    hef_path: str,              # HEF 模型文件路径
    input_queue: queue.Queue,   # 输入队列
    output_queue: queue.Queue,  # 输出队列
    batch_size: int = 1,        # 批处理大小
    input_type: str = None,     # 输入格式类型
    output_type: dict = None,   # 输出格式类型
    send_original_frame: bool = False  # 是否传递原始帧
)
```

## 支持的模型类型

### 目标检测模型
- **YOLO 系列**: YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10
- **YOLOx**: 高性能目标检测
- **SSD**: Single Shot MultiBox Detector
- **CenterNet**: 中心点检测

### 其他支持的任务
- **深度估计**: StereoNet
- **实例分割**: yolov5_seg/yolov8_seg
- **车道检测**: UFLDv2
- **姿态估计**: yolov8 pose
- **语音识别**: Whisper
- **超分辨率**: espcnx4, srgan

## 核心 API 详解

### 1. 设备初始化

```python
# 创建设备参数
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

# 加载 HEF 模型
hef = HEF(hef_path)
target = VDevice(params)
infer_model = target.create_infer_model(hef_path)

# 设置批处理大小
infer_model.set_batch_size(batch_size)
```

### 2. 输入输出格式设置

```python
# 设置输入格式
def _set_input_type(self, input_type: str):
    self.infer_model.input().set_format_type(
        getattr(FormatType, input_type)
    )

# 设置输出格式
def _set_output_type(self, output_type_dict: dict):
    for output_name, output_type in output_type_dict.items():
        self.infer_model.output(output_name).set_format_type(
            getattr(FormatType, output_type)
        )
```

### 3. 异步推理执行

```python
def run(self):
    with self.infer_model.configure() as configured_infer_model:
        while True:
            batch_data = self.input_queue.get()
            if batch_data is None:
                break
            
            # 创建绑定
            bindings_list = []
            for frame in preprocessed_batch:
                bindings = self._create_bindings(configured_infer_model)
                bindings.input().set_buffer(np.array(frame))
                bindings_list.append(bindings)
            
            # 执行异步推理
            configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            job = configured_infer_model.run_async(
                bindings_list, 
                partial(self.callback, input_batch=batch, bindings_list=bindings_list)
            )
            job.wait(10000)
```

### 4. 回调函数处理

```python
def callback(self, completion_info, bindings_list: list, input_batch: list):
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            # 单输出处理
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            # 多输出处理
            else:
                result = {
                    name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                    for name in bindings._output_names
                }
            self.output_queue.put((input_batch[i], result))
```

## 实际应用示例

### 1. 目标检测基础示例

```python
#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_images_opencv
import queue
import threading

def object_detection_example():
    # 初始化队列
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # 创建异步推理实例
    hailo_inference = HailoAsyncInference(
        hef_path="yolov8n.hef",
        input_queue=input_queue,
        output_queue=output_queue,
        batch_size=1,
        send_original_frame=True
    )
    
    # 获取模型输入尺寸
    height, width, _ = hailo_inference.get_input_shape()
    
    # 加载和预处理图像
    images = load_images_opencv("input_image.jpg")
    for image in images:
        # 预处理
        processed = cv2.resize(image, (width, height))
        input_queue.put(([image], [processed]))
    
    input_queue.put(None)  # 结束信号
    
    # 启动推理线程
    inference_thread = threading.Thread(target=hailo_inference.run)
    inference_thread.start()
    
    # 处理结果
    while True:
        result = output_queue.get()
        if result is None:
            break
        original_frame, infer_results = result
        # 处理检测结果...
    
    inference_thread.join()
```

### 2. 实时视频流处理

```python
def real_time_detection():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # 初始化推理
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    hailo_inference = HailoAsyncInference(
        hef_path="yolov8n.hef",
        input_queue=input_queue,
        output_queue=output_queue,
        send_original_frame=True
    )
    
    # 预处理线程
    def preprocess_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预处理帧
            processed = cv2.resize(frame, (width, height))
            input_queue.put(([frame], [processed]))
    
    # 后处理线程
    def postprocess_frames():
        while True:
            result = output_queue.get()
            if result is None:
                break
            
            original_frame, infer_results = result
            # 绘制检测结果
            annotated_frame = draw_detections(original_frame, infer_results)
            cv2.imshow("Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # 启动所有线程
    preprocess_thread = threading.Thread(target=preprocess_frames)
    postprocess_thread = threading.Thread(target=postprocess_frames)
    inference_thread = threading.Thread(target=hailo_inference.run)
    
    preprocess_thread.start()
    postprocess_thread.start()
    inference_thread.start()
    
    # 等待线程结束
    preprocess_thread.join()
    postprocess_thread.join()
    inference_thread.join()
    
    cap.release()
    cv2.destroyAllWindows()
```

### 3. 人脸检测特定实现

```python
def face_detection_embedding():
    """
    人脸检测和特征提取的完整流程
    适用于人脸识别权限控制系统
    """
    
    class FaceDetectionProcessor:
        def __init__(self, hef_path: str):
            self.input_queue = queue.Queue()
            self.output_queue = queue.Queue()
            
            self.hailo_inference = HailoAsyncInference(
                hef_path=hef_path,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                batch_size=1,
                send_original_frame=True
            )
            
        def extract_face_embeddings(self, image: np.ndarray, bbox: tuple):
            """
            从图像中提取人脸区域并生成特征向量
            
            Args:
                image: 输入图像
                bbox: 人脸边界框 (x, y, w, h)
                
            Returns:
                face_embedding: 人脸特征向量
            """
            x, y, w, h = bbox
            face_region = image[y:y+h, x:x+w]
            
            # 预处理人脸区域
            height, width, _ = self.hailo_inference.get_input_shape()
            processed_face = cv2.resize(face_region, (width, height))
            
            # 推理
            self.input_queue.put(([face_region], [processed_face]))
            result = self.output_queue.get()
            
            original_frame, embedding = result
            return embedding
            
        def process_frame_with_faces(self, frame: np.ndarray, face_bboxes: list):
            """
            处理包含多个人脸的帧
            选择面积最大的人脸进行特征提取
            """
            if not face_bboxes:
                return None
                
            # 选择面积最大的人脸
            largest_face = max(face_bboxes, key=lambda bbox: bbox[2] * bbox[3])
            
            # 提取特征
            return self.extract_face_embeddings(frame, largest_face)
```

## 性能优化建议

### 1. 批处理优化
```python
# 使用适当的批处理大小
batch_size = 4  # 根据内存和性能需求调整

# 图像批处理分组
def divide_list_to_batches(images_list: list, batch_size: int):
    for i in range(0, len(images_list), batch_size):
        yield images_list[i:i + batch_size]
```

### 2. 内存管理
```python
# 预分配输出缓冲区
output_buffers = {
    output_info.name: np.empty(
        self.infer_model.output(output_info.name).shape,
        dtype=getattr(np, output_type.lower())
    )
    for output_info in self.hef.get_output_vstream_infos()
}
```

### 3. 多线程优化
```python
# 使用生产者-消费者模式
# 预处理线程：图像加载和预处理
# 推理线程：模型推理
# 后处理线程：结果处理和可视化
```

## 错误处理和调试

### 1. 常见错误处理
```python
try:
    hef = HEF(hef_path)
except Exception as e:
    logger.error(f"Failed to load HEF file: {e}")
    
# 验证输入图像
def validate_images(images: list, batch_size: int):
    if not images:
        raise ValueError('No valid images found')
    if len(images) % batch_size != 0:
        raise ValueError('Images count must be divisible by batch_size')
```

### 2. 调试信息
```python
# 获取模型信息
input_vstream_infos = hef.get_input_vstream_infos()
output_vstream_infos = hef.get_output_vstream_infos()

print(f"Input shape: {input_vstream_infos[0].shape}")
print(f"Output shape: {output_vstream_infos[0].shape}")
```

## 部署要求

### 1. 系统要求
- HailoRT v4.21.0 (推荐版本)
- Python 3.8+
- OpenCV 4.x
- NumPy

### 2. 安装步骤
```bash
# 安装 HailoRT
pip install hailort-X.X.X-cpXX-cpXX-linux_x86_64.whl

# 安装依赖
pip install opencv-python numpy loguru

# 下载和安装 PCIe 驱动（硬件相关）
```

### 3. HEF 模型要求
- 模型必须包含 HailoRT-Postprocess
- 支持的格式：YOLO、SSD、CenterNet 等
- 模型输入格式：通常为 RGB 或 BGR

## 最佳实践

### 1. 项目结构
```
face_rec_project/
├── models/          # HEF 模型文件
├── utils/           # 工具函数
├── processors/      # 处理器类
├── config/          # 配置文件
└── main.py         # 主程序入口
```

### 2. 配置管理
```python
# config.py
class HailoConfig:
    HEF_PATH = "models/face_detection.hef"
    BATCH_SIZE = 1
    INPUT_SIZE = (640, 640)
    CONFIDENCE_THRESHOLD = 0.5
    
    # 摄像头设置
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
```

### 3. 日志管理
```python
from loguru import logger

# 配置日志
logger.add("logs/hailo_inference.log", rotation="1 MB")
logger.info("Hailo inference started")
```

## 结论

本指南提供了基于 Hailo-8 设备的 Python API 完整使用方法，特别适用于人脸识别权限控制系统的开发。通过合理的架构设计和优化策略，可以实现高性能的实时AI推理应用。

关键要点：
- 使用异步推理提高性能
- 合理设计多线程架构
- 注意内存管理和错误处理
- 选择合适的批处理大小
- 确保 HEF 模型兼容性 