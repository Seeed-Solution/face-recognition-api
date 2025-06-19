#!/usr/bin/env python3
"""
FaceEmbed API 服务器启动脚本
"""

import os
import sys

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from app import app
import uvicorn

if __name__ == "__main__":
    # Configuration for high-concurrency deployment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 Starting FaceEmbed API on {host}:{port}")
    print(f"📊 API Documentation: http://{host}:{port}/docs")
    print(f"🏥 Health Check: http://{host}:{port}/health")
    
    # 运行配置（单进程模式，避免workers警告）
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        loop="asyncio",
        http="httptools"
    ) 