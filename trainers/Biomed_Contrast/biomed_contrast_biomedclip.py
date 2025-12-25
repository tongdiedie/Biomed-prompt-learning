from collections import OrderedDict
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
from trainers.prompt_templates import BIOMEDDPT_TEMPLATES
from trainers.prompt_templates import CUSTOM_BIOMEDDPT_TEMPLATES

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer


class TextEncoder(nn.Module):
    """
    文本编码器 - 使用BiomedCLIP的text encoder来编码prompt
    """
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):
        x = self.model.encode_text(prompts, True, tokenized_prompts)
        return x


class PromptLearner(nn.Module):
    """
    可学习的prompt模块
    - 为每个类别学习context vectors
    - 结合BIOMEDDPT_TEMPLATES中的多个描述性prompts
    """
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.BIOMED_CONTRAST.N_CTX
        ctx_init = cfg.TRAINER.BIOMED_CONTRAST.CTX_INIT
        dtype = biomedclip_model.text.transformer.dtype
        ctx_dim = 768
        clip_imsize = 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        
        # 初始化tokenizer
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 初始化context vectors（可学习的prompt部分）
        if ctx_init:
            # 使用给定的单词来初始化context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # 随机初始化
            if cfg.TRAINER.BIOMED_CONTRAST.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        
        # 可学习的context参数
        self.ctx = nn.Parameter(ctx_vectors)

        # 处理类别名称
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        
        # 使用自定义模板格式化prompts
        temp = CUSTOM_BIOMEDDPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]

        # Tokenize prompts
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])
        
        # 创建frozen CLIP用于生成对比学习的目标特征
        biomedclip_model_temp, _ = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        
        # 预计算frozen embeddings（用于知识蒸馏）
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(dtype)
            
            # 为每个类别生成多个template的特征（用于对比学习）
            all_teacher_features = []
            num_temp = cfg.TRAINER.BIOMED_CONTRAST.N_PROMPTS
            
            for i in range(num_temp):
                x_tokenized = torch.cat([
                    self.tokenizer(BIOMEDDPT_TEMPLATES[classname][i]) 
                    for classname in classnames
                ])
                text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        # 存储frozen embeddings - shape: (n_cls, n_prompts, dim)
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        
        # 注册不可学习的buffers
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS token
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS tokens

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.BIOMED_CONTRAST.CLASS_TOKEN_POSITION

    def forward(self):
        """
        构造完整的prompts（prefix + learnable context + suffix）
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            # 格式: [SOS] + [CTX] + [CLASS] + [EOS]
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            # 格式: [SOS] + [CTX_HALF1] + [CLASS] + [CTX_HALF2] + [EOS]
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            # 格式: [SOS] + [CLASS] + [CTX] + [EOS]
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CLIP_Inplanted(nn.Module):
    """
    图像编码器 - 在BiomedCLIP的visual encoder中添加可学习的visual prompts
    这些prompts会被插入到transformer的每一层中
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.text.transformer.dtype
        
        # Visual prompt参数
        self.num_tokens = 4  # 每层添加4个prompt tokens
        self.prompt_dim = 768  # prompt维度
        
        # 浅层prompt（patch embedding之后）
        self.prompt_embeddings = torch.zeros(1, self.num_tokens, self.prompt_dim)
        
        # 深层prompts（每个transformer block）
        self.deep_prompt_embeddings = torch.zeros(12, self.num_tokens, self.prompt_dim)
        
        # Dropout防止过拟合
        self.prompt_dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        在每个transformer layer前插入visual prompts
        """
        # Patch embedding
        x = self.image_encoder.trunk.patch_embed(x)
        x = self.image_encoder.trunk._pos_embed(x)
        x = self.image_encoder.trunk.patch_drop(x)
        x = self.image_encoder.trunk.norm_pre(x)
        
        B = x.shape[0]
        
        # 在CLS token后插入shallow prompts
        x = torch.cat((
            x[:, :1, :],  # CLS token
            self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1).cuda()),  # Prompt tokens
            x[:, 1+self.num_tokens:, :]  # 其余的patch tokens
        ), dim=1)
        
        # 通过12个transformer blocks，每层都添加deep prompts
        for i in range(12):
            B = x.shape[0]
            # 在每层前重新插入prompts
            x = torch.cat((
                x[:, :1, :],  # CLS token
                self.prompt_dropout(self.deep_prompt_embeddings[i].expand(B, -1, -1).cuda()),
                x[:, 1+self.num_tokens:, :]  # 其余tokens
            ), dim=1)
            x = self.image_encoder.trunk.blocks[i](x)
        
        # 最终的norm和projection
        x = self.image_encoder.trunk.norm(x)
        x = x[:, 0]  # 只取CLS token
        x = self.image_encoder.trunk.fc_norm(x)
        x = self.image_encoder.trunk.head_drop(x)
        x = self.image_encoder.trunk.head(x)
        x = self.image_encoder.head(x)
        
        return x

class CustomCLIP(nn.Module):
    """
    自定义CLIP模型 - 结合了:
    1. 可学习的text prompts
    2. 可学习的visual prompts  
    3. 对比学习loss（类别级别）
    """
    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, biomedclip_model)
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = CLIP_Inplanted(biomedclip_model)
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

    def contrastive_loss(self, image_features, text_features, labels):
        """
        类别级对比学习loss
        - 对于每个样本，同类的text embedding是正样本
        - 其他类的text embedding是负样本
        
        Args:
            image_features: (batch_size, dim) - 归一化的图像特征
            text_features: (n_cls, dim) - 归一化的类别文本特征
            labels: (batch_size,) - 样本的类别标签
        
        Returns:
            contrastive_loss: 对比学习损失
        """
        batch_size = image_features.shape[0]
        
        # 计算图像特征和所有类别文本特征的相似度
        # similarity: (batch_size, n_cls)
        similarity = image_features @ text_features.t()
        
        # 创建positive mask: 同类为1，不同类为0
        # labels: (batch_size,) -> (batch_size, 1) -> (batch_size, n_cls)
        labels_expand = labels.unsqueeze(1)  # (batch_size, 1)
        class_indices = torch.arange(self.n_cls).cuda().unsqueeze(0)  # (1, n_cls)
        positive_mask = (labels_expand == class_indices).float()  # (batch_size, n_cls)
        
        # 对比学习: 最大化正样本相似度，最小化负样本相似度
        # 使用InfoNCE loss的变体
        # exp_sim: (batch_size, n_cls)
        exp_sim = torch.exp(similarity / self.cfg.TRAINER.BIOMED_CONTRAST.TEMPERATURE)
        
        # 计算分母: 所有类别的exp similarity之和
        # denominator: (batch_size,)
        denominator = exp_sim.sum(dim=1)
        
        # 计算分子: 正类的exp similarity
        # numerator: (batch_size,)
        numerator = (exp_sim * positive_mask).sum(dim=1)
        
        # 对比损失: -log(正样本概率)
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        return loss.mean()

    def forward(self, image, label=None):
        """
        前向传播 - 训练时返回logits和loss，测试时只返回logits
        """
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # 生成prompts
        prompts = self.prompt_learner()

        # 编码文本和图像
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        # L2归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 获取固定的embeddings（用于蒸馏）
        fixed_embeddings = self.prompt_learner.fixed_embeddings  # (n_cls, n_prompts, dim)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 对多个prompts求平均
        fixed_embeddings = fixed_embeddings.mean(dim=1)  # (n_cls, dim)
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        
        # 计算zero-shot logits（用于KL散度）
        zero_shot_logits = logit_scale * image_features @ fixed_embeddings.cuda().t()
        
        # 计算learned logits
        logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:
            # ========== 训练模式: 计算多个损失 ==========
            
            # 1. 交叉熵损失（分类loss）
            loss_ce = F.cross_entropy(logits, label)
            
            # 2. L1正则化损失（使学习的特征接近固定的特征）
            loss_l1 = F.l1_loss(
                text_features, 
                fixed_embeddings.cuda(), 
                reduction='mean'
            ) * self.cfg.TRAINER.BIOMED_CONTRAST.L1_LAMBDA
            
            # 3. KL散度损失（使learned logits接近zero-shot logits）
            loss_kl = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel() * self.cfg.TRAINER.BIOMED_CONTRAST.KL_LAMBDA
            
            # 4. 对比学习损失（新增 - 核心创新点）
            # 拉近同类距离，拉远不同类距离
            loss_contrast = self.contrastive_loss(
                image_features, 
                text_features, 
                label
            ) * self.cfg.TRAINER.BIOMED_CONTRAST.CONTRAST_LAMBDA
            
            # 总损失
            total_loss = loss_ce + loss_l1 + loss_kl + loss_contrast
            
            return logits, total_loss
        else:
            # 测试模式: 只返回logits
            return logits


@TRAINER_REGISTRY.register()
class BiomedContrast_BiomedCLIP(TrainerX):
    """
    BiomedContrast训练器 - 基于BiomedCLIP的对比学习prompt tuning
    
    主要特点:
    1. 使用可学习的text和visual prompts
    2. 添加类别级对比学习loss
    3. 结合知识蒸馏（L1和KL loss）
    """
    
    def check_cfg(self, cfg):
        """检查配置"""
        assert cfg.TRAINER.BIOMED_CONTRAST.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        """构建模型"""
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        if cfg.TRAINER.BIOMED_CONTRAST.PREC == "fp32" or cfg.TRAINER.BIOMED_CONTRAST.PREC == "amp":
            biomedclip_model.float()

        print("Building custom CLIP with contrastive learning")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())

        print("Turning off gradients in both the image and the text encoder")
        # 只训练prompt learner的context
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)
        
        # 验证哪些参数会被更新
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        
        # 构建优化器和调度器
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        
        # 其他训练设置
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.BIOMED_CONTRAST.PREC == "amp" else None
        
        # 多GPU支持
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        """前向传播和反向传播"""
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.BIOMED_CONTRAST.PREC
        
        if prec == "amp":
            # 混合精度训练
            with autocast():
                logits, loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            # 常规训练
            logits, loss = model(image, label)
            self.model_backward_and_update(loss)

        # 记录损失和准确率
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        # 每个epoch结束时更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        """解析训练batch"""
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        """加载预训练模型"""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # 默认加载最佳模型
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

            # 忽略固定的token vectors（这些是从预训练模型来的）
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # strict=False 允许部分加载
            self._models[name].load_state_dict(state_dict, strict=False)