"""
Weights & Biases 配置文件
用于ControlNet训练实验管理
"""

import wandb

def init_wandb():
    """初始化wandb配置"""
    
    # 设置实验配置
    config = {
        "model": "ControlNet-SD15",
        "dataset": "spatial-ediffsr",
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-5,
            "max_epochs": 10,
            "image_resolution": 256,
            "control_strength": 1.0
        },
        "model_config": {
            "sd_locked": True,
            "only_mid_control": False,
            "control_key": "hint",
            "first_stage_key": "jpg",
            "cond_stage_key": "txt"
        },
        "data": {
            "source_dir": "/workspace/new_data/spatial_ediffsr/tvt/train/2021_new_bicubic8m_clean_png",
            "target_dir": "/workspace/new_data/spatial_ediffsr/tvt/train/2021_new_clean_png",
            "upsample_factor": 4,
            "source_size": "64x64 -> 256x256",
            "target_size": "256x256"
        }
    }
    
    return config

def create_experiment():
    """创建新的wandb实验"""
    
    config = init_wandb()
    
    # 初始化wandb - 使用新的项目名称"controlnet"
    run = wandb.init(
        project="controlnet",  # 项目名称改为"controlnet"
        name="controlnet-lr-dino-pro1,
        config=config,
        tags=["controlnet", "sd15", "spatial-ediffsr", "upsampling", "continue"],
        notes="Continuing ControlNet training for spatial EDiffSR with 4x upsampling",
        save_code=True
    )
    
    return run

if __name__ == "__main__":
    # 测试wandb配置
    try:
        run = create_experiment()
        print("Wandb experiment created successfully!")
        print(f"Project: {run.project}")
        print(f"Run ID: {run.id}")
        print(f"Run URL: {run.url}")
        
        # 记录一些测试数据
        wandb.log({"test/loss": 0.5, "test/step": 0})
        
        run.finish()
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        print("Please check your wandb installation and login status")
