import os
import torch
import numpy as np
import cv2
from PIL import Image
from contextlib import nullcontext
import argparse
from tqdm import tqdm
from pathlib import Path
import random
from einops import rearrange
from pytorch_lightning import seed_everything

from share import *
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.models.autoencoder import AutoencoderKL  # ✅ 加载单独VAE需要


class TestDataset:
    def __init__(self, source_dir, target_dir, image_size=512):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
    
        self.data = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']:
            for source_file in self.source_dir.glob(ext):
                target_file = self.target_dir / source_file.name
                if target_file.exists():
                    self.data.append({
                        'source': source_file,
                        'target': target_file,
                        'filename': source_file.name
                    })
        print(f"找到 {len(self.data)} 对测试图像")
        
    def _upsample_4x(self, image):
        h, w = image.shape[:2]
        return cv2.resize(image, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        source = cv2.imread(str(item['source']))
        target = cv2.imread(str(item['target']))
        
        if source is None or target is None:
            raise ValueError(f"Failed to load: {item['source']}, {item['target']}")
        
        source = self._upsample_4x(source)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        source = cv2.resize(source, (self.image_size, self.image_size))
        target = cv2.resize(target, (self.image_size, self.image_size))
        
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0
        
        return {
            'source': source,
            'target': target,
            'cond_image': (source * 255.0).astype(np.uint8),
            'filename': item['filename']
        }


def load_model(checkpoint_path, model_config_path='./models/cldm_v15.yaml'):
    print(f"正在加载模型: {checkpoint_path}")
    model = create_model(model_config_path).cpu()
    _state = load_state_dict(checkpoint_path, location='cpu')
    _state = {k: v for k, v in _state.items() if not k.startswith('cond_stage_model.')}
    missing, unexpected = model.load_state_dict(_state, strict=False)
    print(f"加载完成 (missing={len(missing)}, unexpected={len(unexpected)})")


    model.cond_stage_key = 'cond_image'
    model.sd_locked = True
    model.only_mid_control = False

    model = model.cuda().eval()
    if hasattr(model, 'cond_stage_model') and model.cond_stage_model is not None:
        try: model.cond_stage_model = model.cond_stage_model.cuda().eval()
        except: pass
    if hasattr(model, 'first_stage_model') and model.first_stage_model is not None:
        try: model.first_stage_model = model.first_stage_model.cuda().eval()
        except: pass
    return model


def inference_single_image(model, ddim_sampler, data, ddim_steps=50, scale=9.0, strength=1.0, seed=-1):
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    source = data['source']
    cond_image = data['cond_image']
    H, W, _ = source.shape

    control = torch.from_numpy(source.copy()).float().cuda()
    control = rearrange(control.unsqueeze(0), 'b h w c -> b c h w').clone()
    print(f"[DEBUG] Control: shape={tuple(control.shape)}, min={control.min().item():.3f}, max={control.max().item():.3f}")

    # DINOv3 encode
    try:
        pil_img = Image.fromarray(cond_image)
        dino_features = model.cond_stage_model.encode([pil_img])
        if isinstance(dino_features, torch.Tensor):
            dino_features = dino_features.to(control.device)
        print(f"[DEBUG] DINO features: shape={tuple(dino_features.shape)}, mean={dino_features.mean().item():.3f}, std={dino_features.std().item():.3f}")
    except Exception as e:
        print(f"[WARN] DINOv3编码失败: {e}")
        dino_features = torch.zeros((1, 196, 768), device=control.device)

    cond = {"c_concat": [control], "c_crossattn": [dino_features]}
    uc_cross = torch.zeros_like(dino_features)
    un_cond = {"c_concat": [control], "c_crossattn": [uc_cross]}

    if hasattr(model, 'control_scales'):
        model.control_scales = [strength] * 13

    shape = (4, H // 8, W // 8)
    print(f"[DEBUG] Latent shape: {shape}")

    ema_ctx = getattr(model, 'ema_scope', None)
    ctx = ema_ctx() if callable(ema_ctx) else nullcontext()
    with ctx:
        samples, _ = ddim_sampler.sample(
            ddim_steps, 1, shape, cond, verbose=True, eta=0.0,
            unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
        )
    print(f"[DEBUG] Samples: min={samples.min().item():.3f}, max={samples.max().item():.3f}, mean={samples.mean().item():.3f}, std={samples.std().item():.3f}")

    x_samples = model.decode_first_stage(samples)
    print(f"[DEBUG] Decoded: min={x_samples.min().item():.3f}, max={x_samples.max().item():.3f}, mean={x_samples.mean().item():.3f}, std={x_samples.std().item():.3f}")

    vis = (rearrange(x_samples, 'b c h w -> b h w c').clamp(-1,1) + 1.0) / 2.0
    return (vis[0].cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/last.ckpt')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./test_results')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--scale', type=float, default=9.0)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint)
    ddim_sampler = DDIMSampler(model)
    dataset = TestDataset(args.source_dir, args.target_dir, args.image_size)

    for i in range(min(3, len(dataset))):  # 只跑前3张图 debug
        data = dataset[i]
        gen = inference_single_image(model, ddim_sampler, data,
                                     args.ddim_steps, args.scale, args.strength, args.seed+i)
        save_path = output_dir / f"debug_{i}_{data['filename']}"
        cv2.imwrite(str(save_path), cv2.cvtColor(gen, cv2.COLOR_RGB2BGR))
        print(f"[INFO] 保存调试图像到 {save_path}")


if __name__ == "__main__":
    main()

# cd ControlNet && conda activate control_1
# python test_metrics.py --checkpoint ./checkpoints/last.ckpt --source_dir /workspace/new_data/spatial_ediffsr/tvt/test/2021_new_bicubic8m_clean_png --target_dir /workspace/new_data/spatial_ediffsr/tvt/test/2021_new_clean_png --output_dir ./test_results --image_size 256 --ddim_steps 50 --scale 9.0 --strength 1.0 --seed 42
