from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from custom_logger import CustomLossLogger, ModelCheckpointWithWandb
import torch


# Configs
resume_path = './checkpoints/last-v1.ckpt'  # 使用最新的checkpoint继续训练
batch_size = 4
logger_freq = 50  # 降低到100步，确保每个epoch都能记录图像
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
# 过滤掉旧CLIP条件分支的权重，并宽松加载
_state = load_state_dict(resume_path, location='cpu')
_state = {k: v for k, v in _state.items() if not k.startswith('cond_stage_model.')}
missing, unexpected = model.load_state_dict(_state, strict=False)
print(f"Loaded state_dict with strict=False. missing={len(missing)}, unexpected={len(unexpected)}")

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control
model.cond_stage_key = 'cond_image'


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

# 创建callbacks
image_logger = ImageLogger(batch_frequency=logger_freq)
custom_loss_logger = CustomLossLogger(log_every_n_steps=50)

# 添加checkpoint保存
checkpoint_callback = ModelCheckpointWithWandb(
    dirpath='./checkpoints',
    filename='controlnet-{epoch:02d}-{step:06d}-{train_loss:.4f}',
    save_top_k=5,  # 保存最好的5个checkpoint
    monitor='train_loss',
    mode='min',
    save_last=True,  
    every_n_train_steps=300  # 每500步保存一次
)


wandb_logger = WandbLogger(
    project="controlnet",  
    name="controlnet-lr",  
    log_model=True,  
    save_dir="./wandb_logs"  
)


accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
trainer = pl.Trainer(
    accelerator=accelerator,
    devices=1,
    precision=32,
    callbacks=[image_logger, custom_loss_logger, checkpoint_callback],
    logger=wandb_logger,
    log_every_n_steps=50,  
    max_epochs=50,  
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=True
)

# 从新训练 or 从checkpoint继续训练
trainer.fit(model, dataloader, ckpt_path=resume_path)


# cd ControlNet && conda activate control_1
# python tutorial_train.py
