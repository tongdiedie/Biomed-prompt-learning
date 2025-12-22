"""
BiomedDPT_Robust (BiomedCLIP backbone)
======================================
BiomedDPT + 低质量 Prompt 鲁棒性增强（BiomedCLIP 版本）

核心改进:
在 L1 损失中添加低质量 Prompt 约束，让模型同时学习：
1. 细粒度语义（从高质量 Prompt）
2. 核心语义（从低质量 Prompt）

损失函数:
L = L_ce + λ1 * L_L1_high + λ2 * L_KL + λ3 * L_L1_low

文件位置：trainers/BiomedDPT_Robust/biomeddpt_robust_biomedclip.py
"""

import copy
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

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer, tokenize



def load_biomedclip_to_cpu(cfg):
    """加载 BiomedCLIP 模型到 CPU"""
    print("Loading BiomedCLIP-PubMedBERT_256-vit_base_patch16_224...")
    clip_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    if cfg.TRAINER.BIOMEDDPT_ROBUST.PREC in ["fp32", "amp"]:
        clip_model.float()
    
    return clip_model, preprocess


class TextEncoder(nn.Module):
    """文本编码器（BiomedCLIP 的 PubMedBERT）"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text.transformer
        self.token_embedding = clip_model.text.token_embedding
        self.positional_embedding = clip_model.text.positional_embedding
        self.ln_final = clip_model.text.ln_final
        self.text_projection = clip_model.text.text_projection
        self.attn_mask = clip_model.text.attn_mask
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """前向传播"""
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 提取 [EOS] token 特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """
    鲁棒性增强的 Prompt 学习器（BiomedCLIP 版本）
    
    包含:
    1. 高质量 Prompt（教师，冻结）：GPT-4 生成
    2. 低质量 Prompt（参考锚点，冻结）：类别名
    3. 可学习 Prompt（学生）：需同时向高质量和低质量对齐
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDDPT_ROBUST.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDDPT_ROBUST.CTX_INIT
        dtype = clip_model.text.transformer.dtype
        ctx_dim = 768
        
        local_path = 'clip\checkpoints\BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir=local_path
        )

        # ========== 1. 初始化可学习 Prompt（学生）==========
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)

            prompt_tokens = self.tokenizer(ctx_init)

            with torch.no_grad():
                embedding = clip_model.text.transformer.embeddings.word_embeddings(
                    prompt_tokens
                ).type(dtype)
            ctx_vectors = embedding[0: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'[INIT] Learnable Prompt: \"{prompt_prefix}\"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        
        # 使用中等质量模板构造可学习 Prompt
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        # tokenized_prompts = torch.cat([tokenize([p], context_length=77) for p in prompts])
        tokenized_prompts = self.tokenizer(prompts)

        # ========== 2. 加载高质量 Prompt（教师，冻结）==========
        print("[TEACHER] Loading high-quality Prompt (GPT-4 generated, frozen)")
        clip_model_temp, _ = load_biomedclip_to_cpu(cfg)
        clip_model_temp = clip_model_temp.float().cuda()
        
        with torch.no_grad():
            embedding = clip_model.text.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(dtype)
            
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            for i in range(cfg.TRAINER.BIOMEDDPT_ROBUST.N_PROMPTS):
                x_tokenized = torch.cat([self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i]) for classname in classnames])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

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
            low_tokenized = self.tokenizer(low_quality_prompts).cuda()
            low_text_features = clip_model_temp.encode_text(low_tokenized, normalize=True)
        
        self.fixed_low_embeddings = low_text_features  # 低质量特征（冻结）
        print(f"[OK] Low-quality Prompt initialized")
        
        # 保存 token 嵌入
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = cfg.TRAINER.BIOMEDDPT_ROBUST.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix):
        """构造完整的 Prompt"""
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """
        返回可学习 Prompt 的嵌入
        
        返回:
            prompts: 可学习 Prompt 嵌入
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CLIP_Inplanted(nn.Module):
    """带 Visual Prompt 的图像编码器（BiomedCLIP 版本）"""
    def __init__(self, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.text.transformer.dtype
        
        # Visual Prompt 参数
        self.num_tokens = 4
        self.prompt_dim = 768  # ViT-B/16 的隐藏维度
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.prompt_dim))
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(12, self.num_tokens, self.prompt_dim))
        self.prompt_dropout = nn.Dropout(0.5)
        
        # 初始化
        nn.init.normal_(self.prompt_embeddings, std=0.02)
        nn.init.normal_(self.deep_prompt_embeddings, std=0.02)

    def forward(self, x):
        """前向传播（注入 Visual Prompt）"""
        # Patch embedding
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 添加 class token
        x = torch.cat([
            self.image_encoder.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)
        
        # 添加 positional embedding
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        # 注入浅层 Visual Prompt
        B = x.shape[0]
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)),
            x[:, 1+self.num_tokens:, :]
        ), dim=1)
        
        # Transformer blocks（注入深层 Visual Prompt）
        for i in range(12):
            B = x.shape[0]
            x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.deep_prompt_embeddings[i].expand(B, -1, -1)),
                x[:, 1+self.num_tokens:, :]
            ), dim=1)
            x = x.permute(1, 0, 2)
            x = self.image_encoder.transformer.resblocks[i](x)
            x = x.permute(1, 0, 2)
        
        # 提取 class token
        x = self.image_encoder.ln_post(x[:, 0, :])
        
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj
        
        return x


class CustomCLIP(nn.Module):
    """鲁棒性增强的 BiomedCLIP 模型"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
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
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # 获取可学习 Prompt
        prompts = self.prompt_learner()

        # 提取特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
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
class BiomedDPT_Robust_BiomedCLIP(TrainerX):
    """BiomedDPT_Robust 训练器（BiomedCLIP backbone）"""
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDDPT_ROBUST.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"\n{'='*80}")
        print(f"Building BiomedDPT_Robust model (BiomedCLIP backbone)")
        print(f"{'='*80}\n")
        
        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model, _ = load_biomedclip_to_cpu(cfg)

        if cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "fp32" or cfg.TRAINER.BIOMEDDPT_ROBUST.PREC == "amp":
            clip_model.float()

        print("构建自定义 BiomedCLIP 模型")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval())

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
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
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

            # 忽略固定的 token 向量（不需要加载）
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
