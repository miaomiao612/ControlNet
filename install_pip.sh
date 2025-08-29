#!/bin/bash

# 使用pip安装DINOv3 + ControlNet环境的简化脚本

echo "🚀 开始安装 DINOv3 + ControlNet 环境..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "📋 Python版本: $python_version"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU模式"
fi

# 升级pip
echo "📦 升级pip..."
python3 -m pip install --upgrade pip

# 安装PyTorch
echo "📦 安装PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # 有GPU，安装CUDA版本
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
else
    # 无GPU，安装CPU版本
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装基础依赖
echo "📦 安装基础依赖..."
pip install numpy pillow matplotlib scikit-image opencv-python

# 安装DINOv3
echo "📦 安装DINOv3..."
pip install dinov3

# 安装ControlNet相关依赖
echo "📦 安装ControlNet相关依赖..."
pip install transformers einops timm gradio albumentations imageio imageio-ffmpeg omegaconf streamlit webdataset kornia invisible-watermark streamlit-drawable-canvas torchmetrics addict yapf prettytable safetensors basicsr

# 安装其他必要依赖
echo "📦 安装其他依赖..."
pip install pytorch-lightning xformers open-clip-torch

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p checkpoints/dinov3
mkdir -p models
mkdir -p annotator/ckpts
mkdir -p output
mkdir -p test_imgs

# 下载DINOv3权重
echo "📥 下载DINOv3权重..."
cd checkpoints/dinov3

# 下载不同大小的模型权重
models=("vit_small" "vit_base" "vit_large")
urls=(
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits14/dinov3_vits14_pretrain.pth"
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb14/dinov3_vitb14_pretrain.pth"
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl14/dinov3_vitl14_pretrain.pth"
)

for i in "${!models[@]}"; do
    model=${models[$i]}
    url=${urls[$i]}
    filename="${model}_pretrain.pth"
    
    if [ ! -f "$filename" ]; then
        echo "下载 $model 权重..."
        wget -O "$filename" "$url"
        echo "✅ $model 权重下载完成"
    else
        echo "✅ $model 权重已存在"
    fi
done

cd ../..

# 测试安装
echo "🧪 测试安装..."
python3 -c "
import torch
import torchvision
print('✅ PyTorch:', torch.__version__)
print('✅ TorchVision:', torchvision.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ CUDA version:', torch.version.cuda)
    print('✅ GPU count:', torch.cuda.device_count())
    print('✅ GPU name:', torch.cuda.get_device_name(0))
"

# 测试DINOv3导入
echo "🧪 测试DINOv3导入..."
python3 -c "
try:
    import dinov3
    print('✅ DINOv3导入成功')
except ImportError as e:
    print('❌ DINOv3导入失败:', e)
"

# 测试自定义模块
echo "🧪 测试自定义模块..."
python3 -c "
try:
    from dinov3_embedder import DINOv3Embedder
    print('✅ DINOv3Embedder导入成功')
except ImportError as e:
    print('❌ DINOv3Embedder导入失败:', e)
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "📋 使用说明:"
echo "1. 运行演示: python3 dinov3_controlnet_demo.py"
echo "2. 查看帮助: python3 dinov3_controlnet_demo.py --help"
echo ""
echo "🔧 可选配置:"
echo "- 下载ControlNet预训练模型到 models/ 目录"
echo "- 下载检测器模型到 annotator/ckpts/ 目录"
echo ""
echo "📚 更多信息请查看 README.md"

