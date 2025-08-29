import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import wandb
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomLossLogger(Callback):
    """自定义loss记录器，用于记录详细的训练指标"""
    
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个训练batch结束后记录loss"""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps == 0:
            # 记录训练loss
            if hasattr(pl_module, 'training_step_outputs'):
                loss = pl_module.training_step_outputs.get('loss', 0.0)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                
                # 记录到wandb
                if trainer.logger:
                    trainer.logger.experiment.log({
                        "train/loss": loss,
                        "train/step": self.step_count,
                        "train/epoch": trainer.current_epoch,
                        "train/learning_rate": pl_module.learning_rate
                    })
                
                # 打印到控制台
                print(f"Step {self.step_count}, Epoch {trainer.current_epoch}, Loss: {loss:.6f}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """在每个epoch结束后记录epoch级别的指标"""
        if trainer.logger:
            # 记录epoch级别的指标
            trainer.logger.experiment.log({
                "train/epoch": trainer.current_epoch,
                "train/epoch_loss": trainer.callback_metrics.get('train_loss', 0.0),
                "train/epoch_step": self.step_count
            })
            
            print(f"Epoch {trainer.current_epoch} completed. Total steps: {self.step_count}")
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """在每个验证batch结束后记录验证loss"""
        if hasattr(pl_module, 'validation_step_outputs'):
            val_loss = pl_module.validation_step_outputs.get('val_loss', 0.0)
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            
            if trainer.logger:
                trainer.logger.experiment.log({
                    "val/loss": val_loss,
                    "val/step": self.step_count,
                    "val/epoch": trainer.current_epoch
                })
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """在每个验证epoch结束后记录验证指标"""
        if trainer.logger:
            trainer.logger.experiment.log({
                "val/epoch": trainer.current_epoch,
                "val/epoch_loss": trainer.callback_metrics.get('val_loss', 0.0)
            })


class ModelCheckpointWithWandb(ModelCheckpoint):
    """扩展的ModelCheckpoint，自动上传到wandb"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """保存checkpoint后自动上传到wandb"""
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            # 上传最新的checkpoint到wandb
            checkpoint_path = self.best_model_path if self.best_model_path else self.last_model_path
            if checkpoint_path:
                try:
                    trainer.logger.experiment.save(checkpoint_path)
                    print(f"Checkpoint uploaded to wandb: {checkpoint_path}")
                except Exception as e:
                    print(f"Failed to upload checkpoint to wandb: {e}")
