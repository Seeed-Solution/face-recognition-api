#!/usr/bin/env python3
"""
运行所有测试的脚本
"""

import os
import sys
import subprocess

def run_tests():
    """运行所有测试"""
    
    # 切换到项目根目录
    project_root = os.path.dirname(os.path.dirname(__file__))
    os.chdir(project_root)
    
    print("🧪 运行FaceEmbed API测试套件...")
    print("=" * 50)
    
    # 运行pytest
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short",
        "--no-header"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 所有测试通过!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败 (exit code: {e.returncode})")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 