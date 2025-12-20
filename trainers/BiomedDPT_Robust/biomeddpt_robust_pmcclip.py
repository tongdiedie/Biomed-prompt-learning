"""
BiomedDPT_Robust (PMC-CLIP backbone)
====================================
BiomedDPT + 低质量 Prompt 鲁棒性增强（PMC-CLIP 版本）

核心改进:
在 L1 损失中添加低质量 Prompt 约束，让模型同时学习：
1. 细粒度语义（从高质量 Prompt）
2. 核心语义（从低质量 Prompt）

损失函数:
L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low

文件位置：trainers/BiomedDPT_Robust/biomeddpt_robust_pmcclip.py
"""

import copy
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

# 导入 Prompt 模板
from trainers.prompt_templates import (
    BIOMEDDPT_TEMPLATES,        # 高质量 GPT-4 Prompt
    CUSTOM_BIOMEDDPT_TEMPLATES, # 中等质量模板
    ZERO_SHOT_TEMPLATES         # 【新增】低质量 Prompt
)

from transformers import AutoTokenizer
import requests
from tqdm import tqdm


def load_pmcclip_to_cpu():
    """加载 PMC-CLIP 模型到 CPU"""
    print("📦 加载 PMC-CLIP (ResNet50) 模型...")
    
    directory = "clip/checkpoints"
    os.makedirs(directory, exist_ok=True)
    
    # PMC-CLIP 模型文件下载链接
    pmcclip_files = {
        "text_encoder.pth": "https://huggingface.co/axiong/pmc_oa_beta/resolve/main/checkpoint.pt",
        "image_encoder(resnet50).pth": "https://huggingface.co/axiong/pmc_oa_beta/resolve/main/model.pth",
        "text_projection_layer.pth": "https://huggingface.co/axiong/pmc_oa_beta/resolve/main/projection.pth"
    }
    
    # 检查并下载模型文件
    for filename, url in pmcclip_files.items():
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f"下载 {filename}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print(f"✅ {filename} 下载完成")
        else:
            print(f"✅ {filename} 已存在")
    
    # 下载 tokenizer
    tokenizer_path = os.path.join(directory, "BiomedNLP-BiomedBERT-base-uncased-abstract")
    if not os.path.exists(tokenizer_path):
        print("下载 BiomedBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            cache_dir=tokenizer_path
        )
    else:
        print("✅ BiomedBERT tokenizer 已存在")
    
    # 构建模型（简化版，实际需要根据 PMC-CLIP 架构调整）
    import torchvision.models as models
    
    class PMCCLIPModel:
        def __init__(self):
            # 图像编码器（ResNet50）
            self.image_encoder = models.resnet50(pretrained=False)
            self.image_encoder.fc = nn.Identity()  # 移除分类层
            image_state_dict = torch.load(
                os.path.join(directory, "image_encoder(resnet50).pth"),
                map_location="cpu"
            )
            self.image_encoder.load_state_dict(image_state_dict)
            
            # 文本编码器（BiomedBERT）
            from transformers import AutoModel
            self.text_encoder = AutoModel.from_pretrained(
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                cache_dir=tokenizer_path
            )
            text_state_dict = torch.load(
                os.path.join(directory, "text_encoder.pth"),
                map_location="cpu"
            )
            self.text_encoder.load_state_dict(text_state_dict)
            
            # 投影层
            self.text_projection = nn.Linear(768, 2048)
            proj_state_dict = torch.load(
                os.path.join(directory, "text_projection_layer.pth"),
                map_location="cpu"
            )
            self.text_projection.load_state_dict(proj_state_dict)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                cache_dir=tokenizer_path
            )
            
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.dtype = torch.float32
        
        def encode_text(self, text_inputs):
            """编码文本"""
            outputs = self.text_encoder(**text_inputs)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            text_features = self.text_projection(text_features)
            return text_features
        
        def encode_image(self, images):
            """编码图像"""
            return self.image_encoder(images)
    
    model = PMCCLIPModel()
    return model


class TextEncoder(nn.Module):
    """文本编码器（PMC-CLIP 的 BiomedBERT）"""
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = clip_model.text_encoder
        self.text_projection = clip_model.text_projection
        self.tokenizer = clip_model.tokenizer
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts=None):
        """
        前向传播
        
        注意：PMC-CLIP 使用 BiomedBERT tokenizer，不同于 CLIP
        """
        # 如果 prompts 是文本列表，先 tokenize
        if isinstance(prompts, list):
            text_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(next(self.text_encoder.parameters()).device)
        else:
            # 如果是预编码的嵌入，直接使用
            text_inputs = {"input_ids": prompts}
        
        # 提取文本特征
        outputs = self.text_encoder(**text_inputs)
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_projection(text_features)
        
        return text_features


class PromptLearner(nn.Module):
    """
    鲁棒性增强的 Prompt 学习器（PMC-CLIP 版本）
    
    包含:
    1. 高质量 Prompt（教师，冻结）：GPT-4 生成
    2. 低质量 Prompt（参考锚点，冻结）：类别名
    3. 可学习 Prompt（学生）：需同时向高质量和低质量对齐
    
    注意：PMC-CLIP 使用 BiomedBERT tokenizer，处理方式不同于 CLIP
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.BIOMEDDPT_ROBUST.N_CTX
        self.dtype = clip_model.dtype
        self.tokenizer = clip_model.tokenizer
        
        # ========== 1. 初始化可学习 Prompt（学生）==========
        ctx_init = cfg.TRAINER.BIOMEDDPT_ROBUST.CTX_INIT
        
        if ctx_init and self.n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init
        else:
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        
        print(f'[INIT] Learnable Prompt: \"{prompt_prefix}\"')
        print(f"上下文长度: {self.n_ctx}")
        
        # 使用中等质量模板构造可学习 Prompt
        classnames = [name.replace("_", " ") for name in classnames]
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        self.prompts_template = [temp.format(c.replace("_", " ")) for c in classnames]
        
        # 对于 PMC-CLIP，我们直接优化文本表示
        # 这里简化为可学习的嵌入向量
        self.ctx = nn.Parameter(torch.randn(self.n_cls, 768, dtype=self.dtype))  # 768 是 BiomedBERT 的隐藏维度
        nn.init.normal_(self.ctx, std=0.02)
        
        # ========== 2. 加载高质量 Prompt（教师，冻结）==========
        print("[TEACHER] Loading high-quality Prompt (GPT-4 generated, frozen)")
        
        with torch.no_grad():
            # 预计算高质量 Prompt 的特征
            all_teacher_features = []
            for i in range(cfg.TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS):
                high_quality_prompts = [
                    BIOMEDDPT_TEMPLATES[classname][i] 
                    for classname in classnames
                ]
                text_inputs = self.tokenizer(
                    high_quality_prompts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt"
                ).to("cuda")
                
                text_features = clip_model.encode_text(text_inputs)
                all_teacher_features.append(text_features.cpu().unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # 高质量特征
        print(f"[OK] High-quality Prompts: {cfg.TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS} per class")
        
        # ========== 3. 【关键新增】初始化低质量 Prompt（鲁棒性锚点，冻结）==========
        print("[ANCHOR] Loading low-quality Prompt (robustness anchor, frozen)")
        low_template_type = cfg.TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE
        
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"警告: 未知模板类型 '{low_template_type}'，使用 'minimal'")
            low_template_type = "minimal"
        
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"低质量模板类型: {low_template_type}")
        
        # 生成低质量 Prompt
        if template == "":
            low_quality_prompts = ["" for _ in classnames]
            print("使用空字符串作为低质量 Prompt")
        else:
            low_quality_prompts = [template.format(**{"class": cls}) for cls in classnames]
            print(f"生成的低质量 Prompt 示例:")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 预计算低质量 Prompt 的特征
        with torch.no_grad():
            text_inputs = self.tokenizer(
                [p if p else "X" for p in low_quality_prompts],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to("cuda")
            
            low_text_features = clip_model.encode_text(text_inputs)
        
        self.fixed_low_embeddings = low_text_features.cpu()  # 低质量特征（冻结）
        print(f"[OK] Low-quality Prompt initialized")

    def forward(self):
        """
        返回可学习 Prompt 的文本列表
        
        返回:
            prompts: 可学习 Prompt 文本列表
        """
        # 返回模板文本（实际训练时会通过 ctx 调整表示）
        return self.prompts_template


class CLIP_Inplanted(nn.Module):
    """带 Visual Prompt 的图像编码器（PMC-CLIP 版本，ResNet50）"""
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.image_encoder
        self.dtype = clip_model.dtype
        
        # Visual Prompt 参数（调整为 ResNet50 的输入维度）
        self.num_tokens = 4
        self.prompt_dim = 2048  # ResNet50 的输出维度
        
        # 注意：对于 ResNet，Visual Prompt 的注入方式需要调整
        # 这里简化为在特征层面添加可学习的偏置
        self.prompt_bias = nn.Parameter(torch.zeros(1, self.prompt_dim))
        nn.init.normal_(self.prompt_bias, std=0.02)

    def forward(self, x):
        """前向传播（ResNet50）"""
        features = self.image_encoder(x)
        
        # 添加可学习的 Visual Prompt（简化版）
        features = features + self.prompt_bias
        
        return features


class CustomCLIP(nn.Module):
    """鲁棒性增强的 PMC-CLIP 模型"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.image_encoder = CLIP_Inplanted(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.cfg = cfg

    def forward(self, image, label=None):
        """
        前向传播
        
        计算损失:
        L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low
        """
        logit_scale = self.logit_scale.exp()

        # 获取可学习 Prompt（文本列表）
        prompts = self.prompt_learner()

        # 提取特征
        text_features = self.text_encoder(prompts)
        
        # 添加可学习的上下文调整
        text_features = text_features + self.prompt_learner.ctx
        
        image_features = self.image_encoder(image.type(self.dtype))
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 高质量特征（教师）
        fixed_embeddings = self.prompt_learner.fixed_embeddings
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 【关键新增】低质量特征（鲁棒性锚点）
        fixed_low_embeddings = self.prompt_learner.fixed_low_embeddings
        fixed_low_embeddings = fixed_low_embeddings / fixed_low_embeddings.norm(dim=-1, keepdim=True)
        
        # 计算 logits
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            # ========== 损失 1：交叉熵损失 ==========
            loss_ce = F.cross_entropy(logits, label)
            
            # ========== 损失 2：L1 对齐损失（可学习 → 高质量）==========
            loss_l1_high = F.l1_loss(
                text_features, 
                fixed_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_HIGH
            
            # ========== 损失 3：KL 散度损失（知识蒸馏）==========
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMEDDPT_ROBUST.KL_LAMBDA
            
            # ========== 【关键新增】损失 4：L1 鲁棒性约束（可学习 → 低质量）==========
            loss_l1_low = F.l1_loss(
                text_features, 
                fixed_low_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW
            
            # ========== 总损失 ==========
            total_loss = loss_ce + loss_l1_high + loss_kl + loss_l1_low
            
            return logits, total_loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class BiomedDPT_Robust_PMCCLIP(TrainerX):
    """BiomedDPT_Robust 训练器（PMC-CLIP backbone）"""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDDPT_ROBUST.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"\n{'='*80}")
        print(f"🚀 构建 BiomedDPT_Robust 模型（PMC-CLIP backbone）")
        print(f"{'='*80}\n")
        
        print(f"加载 PMC-CLIP (ResNet50 + BiomedBERT)")
        clip_model = load_pmcclip_to_cpu()

        print("构建自定义 PMC-CLIP 模型")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("冻结图像和文本编码器，仅优化 Prompt")
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)

        # 检查可训练参数
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"\n[OK] Trainable parameters: {enabled}")
        print(f"✅ 参数数量: {len(enabled)}\n")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"检测到多 GPU ({device_count} 个)，使用全部！")
            self.model = nn.DataParallel(self.model)
        
        print(f"{'='*80}\n")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDDPT_ROBUST.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss = model(image, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
