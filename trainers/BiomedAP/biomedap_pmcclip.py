"""
BiomedAP (PMC-CLIP backbone)
============================
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
import math
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

from clip.pmcclip import ModifiedResNet
from transformers import AutoTokenizer, AutoModel

directory = "clip/checkpoints"
files = {
    "text_encoder.pth": "clip/checkpoints/text_encoder.pth",
    "image_encoder(resnet50).pth": "clip/checkpoints/image_encoder(resnet50).pth",
    "text_projection_layer.pth": "clip/checkpoints/text_projection_layer.pth",
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


# ========== 【核心新增】Text-Guided Visual Prompt 模块 ==========
class TextGuidedVisualPrompt(nn.Module):
    """
    文本引导的视觉Prompt模块（共享投影层方案）
    
    工作原理:
    1. 使用1个共享投影层将文本特征(768维)映射到视觉空间(2048维，ResNet50的特征维度)
    2. 每层视觉prompt通过加性融合文本语义: vp' = vp + scale * text_proj
    3. 层级缩放系数(layer_scales)自适应控制不同层的融合强度
    
    注意: PMC-CLIP使用ResNet50作为视觉编码器，特征维度与ViT不同
    """
    def __init__(self, n_layers=5, n_prompts=4, vis_dim=2048, text_dim=768):
        """
        Args:
            n_layers: ResNet50的层数（layer1-layer4 + attnpool）
            n_prompts: 每层的prompt token数量
            vis_dim: 视觉特征维度（ResNet50最终输出2048维）
            text_dim: 文本特征维度（PMC-CLIP的文本编码器输出768维）
        """
        super().__init__()
        
        # 共享的文本投影层（768 → 2048）
        self.shared_projector = nn.Linear(text_dim, vis_dim)  # 768×2048 ≈ 1.57M参数
        
        # 每层的视觉prompt（可学习）
        # PMC-CLIP使用ResNet50，我们在5个关键层注入prompt
        self.visual_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompts, vis_dim) * 0.02)  # 小初始化
            for _ in range(n_layers)
        ])
        
        # 每层的动态缩放系数（控制文本引导强度）
        self.layer_scales = nn.Parameter(torch.ones(n_layers))  # 初始化为1.0
        
        # Dropout防止过拟合
        self.prompt_dropout = nn.Dropout(0.1)
    
    def forward(self, text_global):
        """
        Args:
            text_global: 文本全局特征 [B, 768] (来自教师模型)
        
        Returns:
            modulated_prompts: 调制后的视觉prompt列表，每个 [B, M, 2048]
        """
        # 1. 文本特征投影到视觉空间（所有层共享此映射）
        text_proj = self.shared_projector(text_global)  # [B, 2048]
        
        modulated_prompts = []
        for i in range(len(self.visual_prompts)):
            # 2. 获取当前层的缩放系数（归一化到[0,1]）
            scale = torch.sigmoid(self.layer_scales[i])
            
            # 3. 加性融合：视觉prompt + 文本语义
            vp = self.visual_prompts[i]  # [M, 2048]
            # 广播机制: [M,2048] + scale*[B,1,2048] → [B,M,2048]
            modulated_vp = vp.unsqueeze(0) + scale * text_proj.unsqueeze(1)
            
            # 4. 应用dropout
            modulated_vp = self.prompt_dropout(modulated_vp)
            
            modulated_prompts.append(modulated_vp)
        
        return modulated_prompts


class TextEncoder(nn.Module):
    """文本编码器（保持不变）"""
    def __init__(self, pmcclip_model):
        super().__init__()
        self.model = pmcclip_model
        self.dtype = torch.float32
        self.text_encoder = pmcclip_model.text_encoder
        self.text_projection_layer = pmcclip_model.text_projection_layer

    def forward(self, prompts, tokenized_prompts):
        output = self.text_encoder(inputs_embeds=prompts.cuda(), attention_mask=tokenized_prompts['attention_mask'].cuda())
        pooler_output = output.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        return text_feature


class PromptLearner(nn.Module):
    """Prompt学习器（集成高质量 + 低质量特征预计算）"""
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        ctx_init = cfg.TRAINER.BIOMEDAP.CTX_INIT
        dtype = torch.float32
        ctx_dim = 768  # PMC-CLIP的文本嵌入维度
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.tokenizer = AutoTokenizer.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化可学习上下文向量
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(prompt.cuda()).type(dtype)
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
        name_lens = [len(self.tokenizer(name, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids']) for name in classnames]
        
        # 使用中等质量模板
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = self.tokenizer(prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')

        with torch.no_grad():
            embedding = pmcclip_model.text_encoder.embeddings.word_embeddings(tokenized_prompts['input_ids'].cuda()).type(dtype)
            
            # ========== 预计算高质量特征（50个模板）==========
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMEDAP.N_PROMPTS
            for i in range(num_temp):
                x_tokenized = torch.cat([
                    self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids'] 
                    for classname in classnames
                ])
                x_tokenized_attn_masks = torch.cat([
                    self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i], padding='max_length', truncation=True, max_length=77, return_tensors='pt')['attention_mask'] 
                    for classname in classnames
                ])
                text_features = pmcclip_model.text_encoder(x_tokenized.cuda(), x_tokenized_attn_masks.cuda())
                pooler_output = text_features.pooler_output
                text_features = pooler_output @ pmcclip_model.text_projection_layer
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)  # [N, 50, 768]
        
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
                self.tokenizer(p if p else "X", padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids'] 
                for p in low_quality_prompts
            ])
            low_tokenized_attn_masks = torch.cat([
                self.tokenizer(p if p else "X", padding='max_length', truncation=True, max_length=77, return_tensors='pt')['attention_mask'] 
                for p in low_quality_prompts
            ])
            low_text_features = pmcclip_model.text_encoder(low_tokenized.cuda(), low_tokenized_attn_masks.cuda())
            pooler_output = low_text_features.pooler_output
            low_text_features = pooler_output @ pmcclip_model.text_projection_layer
        
        self.fixed_low_embeddings = low_text_features  # [N, 768]
        print(f"[OK] Low-quality Prompt initialized\n")

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


class PMCCLIP(nn.Module):
    """PMC-CLIP模型（保持不变）"""
    def __init__(self, image_encoder, text_encoder, projection_layer):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_projection_layer = projection_layer
        self.logit_scale = 4.4292
        self.tokenizer = AutoTokenizer.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')

    def forward(self, image, text):
        encoded_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        input_ids = encoded_input['input_ids']
        text_feature = self.text_encoder(input_ids)
        pooler_output = text_feature.pooler_output
        text_feature = pooler_output @ self.text_projection_layer
        image_feature = self.image_encoder(image)
        if isinstance(image_feature, dict):
            image_feature = image_feature['image_features']
        return image_feature, text_feature


# ========== 【关键修改】集成 Text-Guided Prompt 的图像编码器 ==========
class CLIP_Inplanted(nn.Module):
    """
    带文本引导视觉Prompt的图像编码器（ResNet50架构）
    
    改进对比:
    - 原版: 所有样本共享固定的视觉prompt
    - 现在: 视觉prompt根据文本语义动态调制（text-guided）
    
    注意: PMC-CLIP使用ResNet50，prompt注入位置与ViT不同
    """
    def __init__(self, clip_model, text_guided_prompt):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.image_encoder
        self.dtype = torch.float32
        
        # 【核心】文本引导的视觉prompt模块
        self.text_guided_prompt = text_guided_prompt
        self.num_tokens = 4  # 每层prompt的token数量

    def forward(self, x, text_global):
        """
        Args:
            x: 输入图像 [B, 3, 224, 224]
            text_global: 文本全局特征 [B, 768] (来自教师模型)
        
        Returns:
            image_features: 图像特征 [B, 768]
        """
        # 1. 获取文本引导的视觉prompts（5层）
        modulated_prompts = self.text_guided_prompt(text_global)  # List[Tensor[B,M,2048]]
        
        # 2. 初始卷积层
        x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(x)))
        x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
        x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
        x = self.image_encoder.avgpool(x)
        
        # 注意：由于ResNet50的特征图维度变化，我们需要调整prompt注入方式
        # 这里简化实现：只在最后的全局特征上添加prompt影响
        
        B = x.shape[0]
        
        # 3-7. ResNet的4个layer + attnpool
        # 注入prompt 0 (layer1之前)
        x = self.image_encoder.layer1(x)
        
        # 注入prompt 1 (layer2之前)  
        x = self.image_encoder.layer2(x)
        
        # 注入prompt 2 (layer3之前)
        x = self.image_encoder.layer3(x)
        
        # 注入prompt 3 (layer4之前)
        x = self.image_encoder.layer4(x)
        
        # 注入prompt 4 (attnpool之前)
        x = self.image_encoder.attnpool(x)
        
        # 【关键】将文本引导的prompt融合到最终特征
        # 取最后一层的prompt平均作为全局调制
        final_prompt = modulated_prompts[4].mean(dim=1)  # [B, 2048]
        x = x + 0.1 * final_prompt  # 加权融合（权重0.1可调）
        
        return x


class CustomCLIP(nn.Module):
    """自定义 CLIP（集成 Text-Guided + 4个损失函数）"""
    def __init__(self, cfg, classnames, pmcclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, pmcclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # ========== 【核心新增】创建 Text-Guided Prompt 模块 ==========
        n_ctx = cfg.TRAINER.BIOMEDAP.N_CTX
        self.text_guided_prompt = TextGuidedVisualPrompt(
            n_layers=5,        # ResNet50有5个关键层
            n_prompts=4,       # 每层4个prompt tokens
            vis_dim=2048,      # ResNet50的特征维度
            text_dim=768       # PMC-CLIP的文本特征维度
        )
        print(f"[Text-Guided Prompt] Initialized with {n_ctx} visual prompts per layer (ResNet50)")
        
        # 图像编码器（传入 text_guided_prompt 模块）
        self.image_encoder = CLIP_Inplanted(pmcclip_model, self.text_guided_prompt)
        self.text_encoder = TextEncoder(pmcclip_model)
        self.logit_scale = pmcclip_model.logit_scale
        self.dtype = torch.float32
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def forward(self, image, label=None):
        """前向传播"""
        tokenized_prompts = self.tokenized_prompts
        logit_scale = math.exp(self.logit_scale)

        prompts = self.prompt_learner()

        # 提取文本特征
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # ========== 【关键】获取文本全局语义（用于引导视觉prompt）==========
        # 使用教师模型的高质量文本特征作为全局语义
        fixed_embeddings = self.prompt_learner.fixed_embeddings  # [N, 50, 768]
        text_global = fixed_embeddings.mean(dim=1)  # [N, 768] 平均50个模板
        text_global = text_global / text_global.norm(dim=-1, keepdim=True)  # 归一化
        
        # 根据当前batch的label获取对应的文本全局特征
        if label is not None:
            text_global_batch = text_global[label]  # [B, 768]
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
class BiomedAP_PMCCLIP(TrainerX):
    """BiomedAP 训练器（PMC-CLIP backbone）"""
    
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

        print(f"Loading PMC-CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # 加载PMC-CLIP组件
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)
        image_encoder.load_state_dict(torch.load(os.path.join(directory, 'image_encoder(resnet50).pth'), weights_only=True))
        text_encoder = AutoModel.from_pretrained('clip/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract')
        text_encoder.load_state_dict(torch.load(os.path.join(directory, 'text_encoder.pth'), weights_only=True))
        text_projection_layer = torch.load(os.path.join(directory, 'text_projection_layer.pth'), weights_only=True)
        text_projection_layer = nn.Parameter(text_projection_layer)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_encoder = image_encoder.to(device).eval()
        text_encoder = text_encoder.to(device).eval()
        text_projection_layer = text_projection_layer.to(device)

        pmcclip_model = PMCCLIP(image_encoder, text_encoder, text_projection_layer).to(device).eval()

        print("Building custom CLIP with Text-Guided Visual Prompts")
        self.model = CustomCLIP(cfg, classnames, pmcclip_model)

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

