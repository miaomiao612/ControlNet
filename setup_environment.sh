#!/bin/bash

# DINOv3 + ControlNet 环境配置脚本
# 这个脚本会自动配置支持DINOv3和ControlNet的环境

set -e  # 遇到错误时退出

echo "🚀 开始配置 DINOv3 + ControlNet 环境..."

# 检查系统要求
echo "📋 检查系统要求..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Python版本过低，需要Python 3.8+，当前版本: $python_version"
    exit 1
fi
echo "✅ Python版本: $python_version"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  未检测到NVIDIA GPU，将使用CPU模式"
fi

# 检查conda
if command -v conda &> /dev/null; then
    echo "✅ 检测到conda"
else
    echo "❌ 未检测到conda，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 创建conda环境
echo "🔧 创建conda环境..."
if conda env list | grep -q "control_new"; then
    echo "⚠️  环境 'control_new' 已存在，是否重新创建？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        conda env remove -n control_new
        echo "✅ 已删除旧环境"
    else
        echo "使用现有环境"
    fi
fi

if ! conda env list | grep -q "control_new"; then
    echo "创建新环境 'control_new'..."
    conda env create -f environment.yaml
    echo "✅ 环境创建成功"
fi

# 激活环境
echo "🔌 激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate control_new

# 安装PyTorch (如果conda安装失败)
echo "📦 安装PyTorch..."
if ! python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "通过pip安装PyTorch..."
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
fi

# 安装其他依赖
echo "📦 安装其他依赖..."
pip install -r requirements.txt

# 克隆DINOv3代码库
echo "📥 克隆DINOv3代码库..."
if [ ! -d "dinov3" ]; then
    git clone https://github.com/facebookresearch/dinov3.git
    echo "✅ DINOv3代码库克隆成功"
else
    echo "✅ DINOv3代码库已存在"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p checkpoints/dinov3
mkdir -p models
mkdir -p annotator/ckpts
mkdir -p output

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
python -c "
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
python -c "
import sys
sys.path.append('dinov3')
try:
    from dinov3.models import vision_transformer as vits
    print('✅ DINOv3导入成功')
except ImportError as e:
    print('❌ DINOv3导入失败:', e)
"

# 测试自定义模块
echo "🧪 测试自定义模块..."
python -c "
try:
    from dinov3_embedder import DINOv3Embedder
    print('✅ DINOv3Embedder导入成功')
except ImportError as e:
    print('❌ DINOv3Embedder导入失败:', e)
"

echo ""
echo "🎉 环境配置完成！"
echo ""
echo "📋 使用说明:"
echo "1. 激活环境: conda activate control_new"
echo "2. 运行演示: python dinov3_controlnet_demo.py"
echo "3. 查看帮助: python dinov3_controlnet_demo.py --help"
echo ""
echo "🔧 可选配置:"
echo "- 下载ControlNet预训练模型到 models/ 目录"
echo "- 下载检测器模型到 annotator/ckpts/ 目录"
echo ""
echo "📚 更多信息请查看 README.md"

