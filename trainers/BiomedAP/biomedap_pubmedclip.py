"""
BIOMEDAP (PubMedCLIP backbone)
==============================
Adaptive Prompt Learning with Text-Guided Visual Gating

核心创新:
1. Text-to-Visual Prompt Gating：用文本全局语义动态调制每层视觉prompt
2. 多质量文本蒸馏：高质量（50模板）+ 低质量（单模板）
3. 四重损失优化：CE + L1_high + KL + L1_low

损失函数:
L = L_ce + λ1*L_L1_high + λ2*L_KL + λ3*L_L1_low
"""

# ========== 环境配置 ==========
import os
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import copy
import os.path as osp
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import requests
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import ZERO_SHOT_TEMPLATES

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

directory = "clip/checkpoints"

# File URLs
files = {
    "PubMedCLIP_ViT32.pth": "https://huggingface.co/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true",
}


def download_file(url, filepath):
    """下载模型文件（如果不存在）"""
    print(f"Downloading {filepath}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, "wb") as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"{filepath} downloaded successfully.")
    else:
        print(f"Failed to download {filepath}. HTTP Status Code: {response.status_code}")


def load_clip_to_cpu(cfg):
    """加载 PubMedCLIP 模型到 CPU"""
    backbone_name = "ViT-B/32"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model = clip.build_model(state_dict or model.state_dict())
    checkpoint = torch.load(os.path.join(directory, "PubMedCLIP_ViT32.pth"), weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    return model


# ========== 【核心新增】Text-Guided Visual Prompt 模块 ==========
class TextGuidedVisualPrompt(nn.Module):
    """
    文本引导的视觉Prompt模块（共享投影层方案）
    
    工作原理:
    1. 使用1个共享投影层将文本特征(512维)映射到视觉空间(768维)
    2. 每层视觉prompt通过加性融合文本语义: vp' = vp + scale * text_proj
    3. 层级缩放系数(layer_scales)自适应控制不同层的融合强度
    
    参数量: ~393K (投影层) + 12 (缩放系数) + 12×4×768 (视觉prompt) ≈ 430K
    """
    def __init__(self, n_layers=12, n_prompts=4, dim=768, text_dim=512):
        super().__init__()
        
        # 共享的文本投影层（所有层共用，减少参数量）
        self.shared_projector = nn.Linear(text_dim, dim)  # 512×768 = 393,216参数
        
        # 每层的视觉prompt（可学习）
        self.visual_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompts, dim) * 0.02)  # 小初始化
            for _ in range(n_layers)
        ])
        
        # 每层的动态缩放系数（控制文本引导强度）
        self.layer_scales = nn.Parameter(torch.ones(n_layers))  # 初始化为1.0
        
        # Dropout防止过拟合
        self.prompt_dropout = nn.Dropout(0.1)
    
    def forward(self, text_global):
        """
        Args:
            text_global: 文本全局特征 [B, 512] (来自教师模型)
        
        Returns:
            modulated_prompts: 调制后的视觉prompt列表，每个 [B, M, 768]
        """
        # 1. 文本特征投影到视觉空间（所有层共享此映射）
        text_proj = self.shared_projector(text_global)  # [B, 768]
        
        modulated_prompts = []
        for i in range(len(self.visual_prompts)):
            # 2. 获取当前层的缩放系数（归一化到[0,1]）
            scale = torch.sigmoid(self.layer_scales[i])
            
            # 3. 加性融合：视觉prompt + 文本语义
            vp = self.visual_prompts[i]  # [M, 768]
            # 广播机制: [M,768] + scale*[B,1,768] → [B,M,768]
            modulated_vp = vp.unsqueeze(0) + scale * text_proj.unsqueeze(1)
            
            # 4. 应用dropout
            modulated_vp = self.prompt_dropout(modulated_vp)
            
            modulated_prompts.append(modulated_vp)
        
        return modulated_prompts


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Prompt学习器（集成高质量 + 低质量特征预计算）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDAP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化可学习上下文向量
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # 处理类名
        classnames = [name.replace("_", " ") for name in classnames]
        self.classnames = classnames  # 保存类名（用于低质量prompt生成）
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        # 使用中等质量模板
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        
        # 加载教师模型
        clip_model_temp = load_clip_to_cpu(cfg).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg).float().cuda()
        
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            
            # ========== 预计算高质量特征（50个模板）==========
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDAP.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([
                    clip.tokenize(BIOMEDDPT_TEMPLATES[classname][i]) 
                    for classname in classnames
                ])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # [N, 50, 512]
        
        # ========== 预计算低质量特征（鲁棒性锚点）==========
        print("[ANCHOR] Loading low-quality Prompt (robustness anchor)")
        low_template_type = cfg.TRAINER.BIOMEDAP.LOW_TEMPLATE_TYPE
        
        if low_template_type not in ZERO_SHOT_TEMPLATES:
            print(f"Warning: Unknown template type '{low_template_type}', using 'minimal'")
            low_template_type = "minimal"
        
        template = ZERO_SHOT_TEMPLATES[low_template_type]
        print(f"Low-quality template type: {low_template_type}")
        
        # 生成低质量 Prompt
        if template == "":
            low_quality_prompts = ["X" for _ in classnames]
            print("Using 'X' as low-quality prompt")
        else:
            low_quality_prompts = [
                template.format(**{"class": cls}) for cls in classnames
            ]
            print(f"Low-quality prompt examples (first 3):")
            for cls, prompt in zip(classnames[:3], low_quality_prompts[:3]):
                print(f"  {cls:15} -> '{prompt}'")
        
        # 预计算低质量特征
        with torch.no_grad():
            low_tokenized = torch.cat([
                clip.tokenize(p if p else "X") for p in low_quality_prompts
            ])
            low_text_features = clip_model_temp.encode_text(low_tokenized.cuda())
        
        self.fixed_low_embeddings = low_text_features  # [N, 512]
        print(f"[OK] Low-quality Prompt initialized\n")
        
        del clip_model_temp  # 释放内存
        del clip_model_temp_image

        # 保存 token 前缀和后缀
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        """构造完整的Prompt"""
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self):
        """前向传播"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts


# ========== 【关键修改】集成 Text-Guided Prompt 的图像编码器 ==========
class CLIP_Inplanted(nn.Module):
    """
    带文本引导视觉Prompt的图像编码器
    
    改进对比:
    - 原版: 所有样本共享固定的视觉prompt
    - 现在: 视觉prompt根据文本语义动态调制（text-guided）
    """
    def __init__(self, clip_model, text_guided_prompt):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        
        # 【核心】文本引导的视觉prompt模块
        self.text_guided_prompt = text_guided_prompt
        self.num_tokens = 4  # 每层prompt的token数量

    def forward(self, x, text_global):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
            text_global: 文本全局特征 [B, 512] (来自教师模型)
        
        Returns:
            image_features: 图像特征 [B, 512]
        """
        # 1. 获取文本引导的视觉prompts（每层动态调制）
        modulated_prompts = self.text_guided_prompt(text_global)  # List[Tensor[B,M,768]]
        
        # 2. Patch embedding
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.image_encoder.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        # 3. 注入第0层的文本引导prompt（浅层特征）
        B = x.shape[0]
        shallow_prompt = modulated_prompts[0]  # [B, M, 768]
        x = torch.cat((
            x[:, :1, :],           # CLS token
            shallow_prompt,        # 文本引导的浅层prompt
            x[:, 1:, :]            # 其他patch tokens
        ), dim=1)
        
        # 4. Transformer blocks（每层注入对应的文本引导prompt）
        for i in range(12):
            deep_prompt = modulated_prompts[i]  # [B, M, 768]
            x = torch.cat((
                x[:, :1, :],                         # CLS token
                deep_prompt,                         # 文本引导的深层prompt
                x[:, 1+self.num_tokens:, :]          # 其他tokens（去掉上一层的prompt）
            ), dim=1)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.image_encoder.transformer.resblocks[i](x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 5. 最终投影
        x = self.image_encoder.ln_post(x[:, 0, :])
        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj
        return x


class CustomCLIP(nn.Module):
    """自定义 CLIP 模型（集成 Text-Guided + 4个损失函数）"""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.cfg = cfg
        
        # ========== 【核心新增】创建 Text-Guided Prompt 模块 ==========
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        self.text_guided_prompt = TextGuidedVisualPrompt(
            n_layers=12,       # ViT-B/32 有 12 层
            n_prompts=4,       # 每层 4 个 prompt tokens
            dim=768,           # ViT-B/32 的 hidden_dim
            text_dim=512       # PubMedCLIP 的文本特征维度
        )
        print(f"[Text-Guided Prompt] Initialized with {n_ctx} visual prompts per layer")
        
        # 图像编码器（传入 text_guided_prompt 模块）
        self.image_encoder = CLIP_Inplanted(clip_model, self.text_guided_prompt)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        """前向传播"""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        # 提取文本特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # ========== 【关键】获取文本全局语义（用于引导视觉prompt）==========
        # 使用教师模型的高质量文本特征作为全局语义
        fixed_embeddings = self.prompt_learner.fixed_embeddings  # [N, 50, 512]
        text_global = fixed_embeddings.mean(dim=1)  # [N, 512] 平均50个模板
        text_global = text_global / text_global.norm(dim=-1, keepdim=True)  # 归一化
        
        # 根据当前batch的label获取对应的文本全局特征
        if label is not None:
            text_global_batch = text_global[label]  # [B, 512]
        else:
            # 测试时使用所有类的平均
            text_global_batch = text_global.mean(dim=0, keepdim=True).expand(image.size(0), -1)
        
        # 提取图像特征（传入文本全局语义）
        image_features = self.image_encoder(image.type(self.dtype), text_global_batch.cuda())
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 高质量特征（教师）
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.mean(dim=1)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 低质量特征（鲁棒性锚点）
        fixed_low_embeddings = self.prompt_learner.fixed_low_embeddings
        fixed_low_embeddings = fixed_low_embeddings / fixed_low_embeddings.norm(dim=-1, keepdim=True)
        
        # 计算 logits
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        logits = logit_scale * image_features @ text_features.t()
        
        # ========== 训练模式：计算4项损失 ==========
        if self.prompt_learner.training:
            # 损失 1: 交叉熵
            loss_ce = F.cross_entropy(logits, label)
            
            # 损失 2: L1 对齐（可学习 → 高质量）
            loss_l1_high = F.l1_loss(
                text_features, 
                fixed_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_HIGH
            
            # 损失 3: KL 散度（知识蒸馏）
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMEDAP.KL_LAMBDA

            # 损失 4: L1 鲁棒性约束（可学习 → 低质量）
            loss_l1_low = F.l1_loss(
                text_features, 
                fixed_low_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMEDAP.L1_LAMBDA_LOW

            # 总损失（4项）
            total_loss = loss_ce + loss_l1_high + loss_kl + loss_l1_low
            
            return logits, total_loss
        else:
            return logits


@TRAINER_REGISTRY.register()
class BIOMEDAP_PubMedCLIP(TrainerX):
    """BiomedAP 训练器（PubMedCLIP backbone）"""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # 下载模型文件（如果不存在）
        for filename, url in files.items():
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                print(f"{filename} not found in {directory}. Downloading...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists in {directory}.")

        print(f"Loading PubMedCLIP (backbone: ViT-B/32)")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BIOMEDAP.PREC == "fp32" or cfg.TRAINER.BIOMEDAP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP with Text-Guided Visual Prompts")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        # 【关键】只优化文本prompt和text-guided模块
        names_to_update = [
            "prompt_learner.ctx",              # 文本prompt
            "text_guided_prompt"                # Text-guided模块（全部参数）
        ]

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)  # 先全部冻结
            for update_name in names_to_update:
                if update_name in name:
                    param.requires_grad_(True)  # 解冻需要更新的
                    break

        # 检查可训练参数
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        print(f"Total trainable parameters: {len(enabled)}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameter count: {total_params:,}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDAP.PREC == "amp" else None
        
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """前向和反向传播"""
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMEDAP.PREC
        if prec == "amp":
            with autocast():
                logits, loss = model(image, label)
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
        """解析训练批次"""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """加载模型"""
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
                print(f"No pretrained model found at '{model_path}', training from scratch")
                return

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 忽略固定的 token 向量
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
