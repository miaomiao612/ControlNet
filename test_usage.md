# ControlNet 测试脚本使用说明

## 功能
这个脚本用于测试训练好的ControlNet模型，计算PSNR、SSIM、LPIPS、FID等评估指标。

## 安装依赖
```bash
pip install scikit-image lpips pytorch-fid
```

## 使用方法

### 基本用法
```bash
python test_metrics.py \
    --checkpoint ./checkpoints/last.ckpt \
    --source_dir /path/to/source/images \
    --target_dir /path/to/target/images \
    --output_dir ./test_results
```

### 参数说明
- `--checkpoint`: 训练好的模型checkpoint路径
- `--source_dir`: 源图像目录（低分辨率图像）
- `--target_dir`: 目标图像目录（高分辨率GT图像）
- `--output_dir`: 输出目录，保存生成图像和结果
- `--image_size`: 图像尺寸，默认512
- `--ddim_steps`: DDIM采样步数，默认50
- `--scale`: CFG scale，默认9.0
- `--strength`: Control strength，默认1.0
- `--seed`: 随机种子，默认42

### 示例
```bash
# 使用默认参数测试
python test_metrics.py \
    --source_dir /workspace/new_data/spatial_ediffsr/tvt/test/2021_new_bicubic8m_clean_png \
    --target_dir /workspace/new_data/spatial_ediffsr/tvt/test/2021_new_clean_png \
    --output_dir ./test_results

# 自定义参数测试
python test_metrics.py \
    --checkpoint ./checkpoints/controlnet-epoch=10-step=5000-train_loss=0.1234.ckpt \
    --source_dir /workspace/test_data/low_res \
    --target_dir /workspace/test_data/high_res \
    --output_dir ./test_results_custom \
    --image_size 256 \
    --ddim_steps 100 \
    --scale 7.5 \
    --strength 0.8 \
    --seed 123
```

## 输出结果

### 文件结构
```
test_results/
├── generated_image1.png
├── generated_image2.png
├── ...
├── test_results.json
├── real_images/     # 用于FID计算的真实图像
└── fake_images/     # 用于FID计算的生成图像
```

### 结果文件格式
`test_results.json` 包含：
- 平均指标（PSNR、SSIM、LPIPS、FID）
- 每张图像的详细指标
- 测试配置参数

### 示例结果
```json
{
  "average_metrics": {
    "psnr": 28.45,
    "ssim": 0.8234,
    "lpips": 0.1567,
    "fid": 45.23
  },
  "individual_metrics": {
    "image1.png": {
      "psnr": 29.12,
      "ssim": 0.8345,
      "lpips": 0.1432
    }
  },
  "test_config": {
    "checkpoint": "./checkpoints/last.ckpt",
    "source_dir": "/path/to/source",
    "target_dir": "/path/to/target",
    "output_dir": "./test_results",
    "image_size": 512,
    "ddim_steps": 50,
    "scale": 9.0,
    "strength": 1.0,
    "seed": 42
  }
}
```

## 注意事项

1. **数据格式**: 脚本会自动处理图像格式转换和归一化
2. **内存使用**: 大量图像测试时注意GPU内存使用
3. **FID计算**: 需要安装pytorch-fid，计算时间较长
4. **图像匹配**: 确保source_dir和target_dir中的图像文件名一致

## 故障排除

### 常见错误
1. **ModuleNotFoundError**: 安装缺失的依赖包
2. **CUDA out of memory**: 减少batch_size或image_size
3. **FID计算失败**: 检查pytorch-fid安装和图像数量

### 调试模式
可以修改脚本添加更多调试信息：
```python
# 在inference_single_image函数中添加
print(f"Processing {filename}: source_shape={source.shape}, target_shape={target.shape}")
```
