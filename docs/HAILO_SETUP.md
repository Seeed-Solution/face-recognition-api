# Hailo-8 必需依赖安装指南

本指南详细说明如何在Raspberry Pi 5上安装和配置Hailo-8 AI加速器。

**重要提示**: FaceEmbed API 服务需要Hailo-8硬件支持，没有Hailo环境将无法启动。

## 🔧 硬件要求

- **Raspberry Pi 5** (推荐8GB内存版本)
- **Hailo-8 AI Kit** (M.2接口)
- **M.2 HAT** 或兼容的PCIe扩展板
- **充足的电源** (推荐5V/5A)

## 📥 软件下载

### 1. 从官方网站下载HailoRT

访问 [Hailo开发者中心](https://hailo.ai/developer-zone/software-downloads/) 下载以下文件：

**必需文件（版本4.21.0）:**
- `hailort_4.21.0_arm64.deb` - 系统包
- `hailort-pcie-driver_4.21.0_all.deb` - PCIe驱动
- `hailort-4.21.0-cp311-cp311-linux_aarch64.whl` - Python包

> 💡 **提示**: 需要注册Hailo开发者账号才能下载

## 🛠️ 系统配置

### 1. 更新系统

```bash
# 同步系统时间
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"

# 更新系统
sudo apt update
sudo apt full-upgrade
```

### 2. 安装系统依赖

```bash
# 安装必要的系统包
sudo apt install -y \
    dkms \
    build-essential \
    linux-headers-$(uname -r) \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0
```

### 3. 安装HailoRT系统包

```bash
# 安装HailoRT系统包
sudo dpkg -i hailort_4.21.0_arm64.deb

# 安装PCIe驱动
sudo dpkg -i hailort-pcie-driver_4.21.0_all.deb
```

### 4. 配置PCIe设置

```bash
# 配置PCIe页面大小
echo "options hailo_pci force_desc_page_size=4096" | sudo tee /etc/modprobe.d/hailo_pci.conf

# 设置PCIe速度为Gen3 (可选，更快性能)
echo "options pcie_aspm policy=performance" | sudo tee /etc/modprobe.d/pcie_performance.conf
```

### 5. 重启系统

```bash
sudo reboot
```

## 🐍 Python环境配置

### 1. 安装HailoRT Python包

```bash
# 激活项目虚拟环境
source .venv/bin/activate

# 安装HailoRT Python包
pip install hailort-4.21.0-cp311-cp311-linux_aarch64.whl
```

### 2. 验证安装

```bash
# 检查Hailo设备
python -c "
try:
    import hailo_platform as hp
    devices = hp.Device.scan()
    print(f'Found {len(devices)} Hailo device(s): {devices}')
except Exception as e:
    print(f'Error: {e}')
"
```

预期输出：
```
Found 1 Hailo device(s): ['0001:01:00.0']
```

## 🔍 故障排除

### 检查PCIe连接

```bash
# 查看PCIe设备
lspci | grep Hailo

# 检查驱动状态
lsmod | grep hailo

# 查看设备状态
dmesg | grep -i hailo
```

### 检查性能

```bash
# 检查PCIe链路速度
sudo lspci -vv | grep -A5 "Hailo"

# 应该显示 "LnkSta: Speed 8GT/s, Width x1" (Gen3)
# 或 "LnkSta: Speed 5GT/s, Width x1" (Gen2)
```

### 常见问题

**1. 设备未识别**
```bash
# 检查硬件连接
sudo dmesg | tail -20

# 重新加载驱动
sudo modprobe -r hailo_pci
sudo modprobe hailo_pci
```

**2. 权限问题**
```bash
# 添加用户到hailo组
sudo usermod -a -G hailo $USER

# 重新登录或使用
newgrp hailo
```

**3. 内存不足**
```bash
# 检查内存使用
free -h

# 增加swap（如果需要）
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 📊 性能优化

### PCIe性能设置

```bash
# 禁用PCIe电源管理（提高性能）
echo "performance" | sudo tee /sys/module/pcie_aspm/parameters/policy

# 设置CPU性能模式
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 系统优化

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

## 🧪 测试安装

### 1. 测试FaceEmbed API

```bash
# 启动服务
python scripts/start_server.py

# 在新终端测试
python scripts/test_hailo_request.py
```

### 2. 性能基准测试

如果看到以下输出，说明Hailo硬件正常工作：
```
Status: 200
Processing time: 3ms          # <- 硬件加速的低延迟
Confidence: 0.7
Vector length: 512
Vector norm: 1.000000
```

**注意**: 如果服务无法启动或显示导入错误，说明HailoRT安装不完整。

## 📚 参考资料

- [Hailo官方文档](https://hailo.ai/developer-zone/)
- [Raspberry Pi 5 + Hailo-8 基准测试](https://wiki.seeedstudio.com/benchmark_of_multistream_inference_on_raspberrypi5_with_hailo8/)
- [HailoRT API文档](https://hailo.ai/developer-zone/documentation/)

## 🆘 获取帮助

如果遇到问题：

1. 检查 [故障排除](#故障排除) 部分
2. 确认硬件连接正确
3. 验证所有依赖包安装完成
4. 查看项目 [GitHub Issues](https://github.com/your-repo/issues)
5. 访问 [Hailo社区论坛](https://community.hailo.ai/)

---

*最后更新: 2024年6月6日* 