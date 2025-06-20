
> **HailoRT** 是 Hailo 设备（如 Hailo-8 加速棒）的“操作系统”，负责和硬件通信、加载模型、执行推理等。
> 
> 没有 HailoRT，其他软件（无论是 Python、C++ 还是 Docker 容器）都无法直接调用 Hailo 加速棒进行 AI 推理。

## HailoRT 所需文件

在本项目中，我们会用到三个文件来进行安装：

- `hailort_<version>_<arch>.deb` - HailoRT 安装包
- `hailort-pcie-driver_<version>_all.deb` - HailoRT PCle 驱动，通过 PCle 连接的 Hailo 设备需要此驱动
- `hailort-<version>-<python-tag>-<abi-tag>-<platform-tag>.whl` - PyHailoRT，通过 Python API 来访问 Hailo 功能

在 [Software Downloads - Developer Zone](https://hailo.ai/developer-zone/software-downloads/) 下载所需要的软件版本（需要注册账号并登录才能访问）可以下载到这些文件。

### 安装 DKMS

因为 HailoRT 依赖的 PCIe 驱动需要针对当前运行内核的版本动态编译和加载，所以需要安装 DKMS。

```bash
sudo apt-get install dkms
```

### 安装 HailoRT

安装 HailoRT、PCle 驱动：

```bash
hailort_version=4.21.0 

sudo dpkg --install hailort_${hailort_version}_$(dpkg --print-architecture).deb hailort-pcie-driver_${hailort_version}_all.deb
```

安装完后，你还需要在 `/etc/modprobe.d/hailo_pci.conf` 添加一行配置项。

```shell
echo "options hailo_pci force_desc_page_size=4096" | sudo tee /etc/modprobe.d/hailo_pci.conf
```

然后，重启设备

```shell
sudo reboot
```

### 检查安装是否成功

你可以用 [hailortcli](https://hailo.ai/developer-zone/documentation/hailort-v4-21-0/file/?page=cli%2Fcli.html) 检查设备是否被识别：

```shell
hailortcli scan
```

这个扫描命令应该能找到设备。

再运行下面的命令，从设备获取序列号：

```shell
hailortcli fw-control identify
```

出现如下类似的信息，说明你的驱动安装成功了：

```shell
hailortcli fw-control identify

Executing on device: 0001:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.21.0 (release,app,extended context switch buffer)
Logger Version: 0
Board Name: Hailo-8
Device Architecture: HAILO8
Serial Number: <N/A>
Part Number: <N/A>
Product Name: <N/A>
```