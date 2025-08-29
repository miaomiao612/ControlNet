import sys
sys.path.append("/workspace/dinov3")   # 直接告诉 Python 去 /workspace/dinov3 里找
from dinov3.models import vision_transformer as vits
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from PIL import Image
import os
#import open_clip
import numpy as np
from ldm.util import default, count_params

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""
    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]
    def __init__(self, version="/workspace/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer=="hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenDINOv3Embedder(AbstractEncoder):
    """Use DINOv3 features as condition encoder"""
    def __init__(self,
                 backbone_ckpt="/workspace/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
                 repo_dir="/workspace/dinov3",
                 model_name="vit_large",
                 patch_size=16,
                 out_dim=768,
                 align_dim=768,
                 device="cuda"):
        super().__init__()
        import sys, os
        sys.path.insert(0, repo_dir)
       
    

        if model_name == "vit_large":
            self.model = vits.vit_large(patch_size=patch_size, num_classes=0)
        elif model_name == "vit_base":
            self.model = vits.vit_base(patch_size=patch_size, num_classes=0)
        elif model_name == "vit_small":
            self.model = vits.vit_small(patch_size=patch_size, num_classes=0)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")

        # 加载权重
        if os.path.exists(backbone_ckpt):
            state_dict = torch.load(backbone_ckpt, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ 成功加载DINOv3权重: {backbone_ckpt}")
        
        self.model.eval().to(device)
        self.device = device

        self.proj = nn.Linear(self.model.embed_dim, out_dim).to(device) 
        print(f"DINOv3权重加载完成，输出维度: {out_dim}")

        # # MLP：768 -> 768（带非线性），让分布更接近 CLIP
        # self.align_proj = nn.Sequential(
        #     nn.Linear(out_dim, align_dim),
        #     nn.GELU(),
        #     nn.Linear(align_dim, align_dim)
        # ).to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.430, 0.411, 0.296),
                                 std=(0.213, 0.156, 0.143)),
        ])

    def forward(self, images):
        if not torch.is_tensor(images):
            images = torch.stack([self.preprocess(img) for img in images])
        images = images.to(self.device)

        with torch.no_grad():
            feats = self.model.get_intermediate_layers(images, n=1)[0]  # [B, N, 1024]
        feats = feats.to(self.device)
        feats = self.proj(feats)
        # feats = self.align_proj(feats)
        return feats

    def encode(self, images):
        return self(images)

