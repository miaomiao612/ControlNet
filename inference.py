import os
import argparse
import torch
import numpy as np
from PIL import Image
from glob import glob
import cv2

from cldm.model import create_model, load_state_dict


# ========= 图像预处理 =========
def upsample_4x(image: np.ndarray) -> np.ndarray:
    """4倍上采样，和训练时一致"""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


def load_image(path: str, image_size: int) -> np.ndarray:
    """加载图像 (RGB, uint8)，并resize到指定大小"""
    img = Image.open(path).convert('RGB')
    img = np.array(img)
    img = upsample_4x(img) 
    if image_size is not None and image_size > 0:
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return img


def prepare_control_numpy(arr: np.ndarray) -> torch.Tensor:
    """转为 BHWC float32 [0,1]"""
    arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr)


def save_tensor_image(x: torch.Tensor, out_path: str):
    """保存 [-1,1] BCHW tensor 到 PNG"""
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.mul(255).add(0.5).clamp(0, 255).byte()
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    Image.fromarray(x).save(out_path)


# ========= 推理 =========
def infer_single(model, cond_image_path, control_image_path, args, out_path):
    # ---- 加载图像 ----
    cond_np = load_image(cond_image_path, args.image_size)       # uint8, HWC
    control_np = load_image(control_image_path, args.image_size) # uint8, HWC

    # ControlNet hint: float32 [0,1]
    control_bhwc = prepare_control_numpy(control_np).to(args.device)
    print(f"[DEBUG] Control: shape={tuple(control_bhwc.shape)}, "
          f"min={control_bhwc.min().item():.3f}, max={control_bhwc.max().item():.3f}")

    # DINO 输入: uint8 (上采样+resize 后)
    try:
        cond_pil = Image.fromarray(cond_np.astype(np.uint8))
        with torch.no_grad():
            dino_feat = model.cond_stage_model.encode([cond_pil])
        if isinstance(dino_feat, torch.Tensor):
            dino_feat = dino_feat.to(args.device)
        else:
            dino_feat = torch.as_tensor(dino_feat, device=args.device)

        print(f"[DEBUG] DINO features: shape={tuple(dino_feat.shape)}, "
              f"mean={dino_feat.mean().item():.3f}, std={dino_feat.std().item():.3f}")
    except Exception as err:
        print(f"[ERROR] DINO encode failed: {err}. Using zeros.")
        # fallback: 假设 DINO 特征是 [1, 196, 768]
        dino_feat = torch.zeros((1, 196, 768),
                                device=args.device, dtype=torch.float32)

    # ---- 构造条件 ----
    c_cat = control_bhwc.permute(0, 3, 1, 2).contiguous()  # BCHW
    cond = {'c_concat': [c_cat], 'c_crossattn': [dino_feat]}
    un_cond = {'c_concat': [torch.zeros_like(c_cat)],
               'c_crossattn': [torch.zeros_like(dino_feat)]}

    # ---- 采样 ----
    with torch.no_grad():
        context_mgr = getattr(model, 'ema_scope', None)
        if callable(context_mgr):
            with model.ema_scope("inference"):
                samples, _ = model.sample_log(
                    cond=cond, batch_size=1, ddim=True,
                    ddim_steps=args.ddim_steps, eta=args.eta,
                    unconditional_guidance_scale=args.scale,
                    unconditional_conditioning=un_cond
                )
        else:
            samples, _ = model.sample_log(
                cond=cond, batch_size=1, ddim=True,
                ddim_steps=args.ddim_steps, eta=args.eta,
                unconditional_guidance_scale=args.scale,
                unconditional_conditioning=un_cond
            )

        x = model.decode_first_stage(samples)
        print(f"[DEBUG] Decoded image: shape={tuple(x.shape)}, "
              f"mean={x.mean().item():.3f}, std={x.std().item():.3f}")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    save_tensor_image(x, out_path)
    print(f"[INFO] Saved: {out_path}")


# ========= 主入口 =========
def main():
    parser = argparse.ArgumentParser(description="Batch inference with ControlNet + DINOv3")
    parser.add_argument('--config', type=str, default='models/cldm_v15.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cond_image', type=str, required=True, help='单张cond图/文件夹 (文件夹时自动批量)')
    parser.add_argument('--control_image', type=str, help='单张control图 (批量时默认与cond相同)')
    parser.add_argument('--output', type=str, default='./inference_out.png')
    parser.add_argument('--output_dir', type=str, default='./inference_outs')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=9.0, help='CFG 总 scale')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    # 固定随机数种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # 加载模型
    model = create_model(args.config).to(args.device)
    state_dict = load_state_dict(args.checkpoint, location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 批量 or 单张
    if os.path.isdir(args.cond_image):
        os.makedirs(args.output_dir, exist_ok=True)
        exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
        files = []
        for e in exts:
            files.extend(glob(os.path.join(args.cond_image, e)))
        files = sorted(files)
        print(f"Found {len(files)} images in {args.cond_image}")

        for img_path in files:
            fname = os.path.basename(img_path)
            out_path = os.path.join(args.output_dir, fname)
            infer_single(model, img_path, img_path, args, out_path)
    else:
        infer_single(model, args.cond_image, args.control_image or args.cond_image, args, args.output)


if __name__ == '__main__':
    main()

# cd ControlNet && conda activate control_1
# python inference.py --config models/cldm_v15.yaml --checkpoint checkpoints/controlnet_dinov3_last49.ckpt --cond_image /workspace/new_data/spatial_ediffsr/tvt/train/2021_new_bicubic8m_clean_png --control_image /workspace/new_data/spatial_ediffsr/tvt/train/2021_new_bicubic8m_clean_png --output_dir ./inference_outs --image_size 256 --ddim_steps 50 --scale 9.0 --seed 42
