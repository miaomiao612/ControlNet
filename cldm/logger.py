import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only  # PL < 2.0
except ImportError:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only   # PL >= 2.0


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if isinstance(images[k], torch.Tensor):
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # [-1,1] -> [0,1]
                grid = grid.transpose(0, 1).transpose(1, 2).cpu().numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx
        if (self.check_frequency(check_idx)
                and hasattr(pl_module, "log_images")
                and callable(pl_module.log_images)
                and self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # 统一清洗输出
            for k in list(images.keys()):
                val = images[k]
                if isinstance(val, torch.Tensor):
                    N = min(val.shape[0], self.max_images)
                    val = val[:N].detach().cpu()
                    if self.clamp:
                        val = torch.clamp(val, -1., 1.)
                    images[k] = val
                elif isinstance(val, list) and all(isinstance(x, str) for x in val):
                    # 如果是字符串（例如 text conditioning），跳过保存
                    images.pop(k)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

            return images  
        return None

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.disabled:
            return
        if (batch_idx % self.batch_freq == 0) or (self.log_first_step and batch_idx == 0):
            imgs = self.log_img(pl_module, batch, batch_idx, **self.log_images_kwargs)
            if imgs is not None:
                print(f"[ImageLogger] logged {len(imgs)} items at step {trainer.global_step}")
