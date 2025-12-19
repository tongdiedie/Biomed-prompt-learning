#!/bin/bash

# ========================================
# BiomedDPT_Robust Few-Shot 训练脚本
# 文件位置：scripts/few_shot_robust.sh
#
# 用法：
# bash scripts/few_shot_robust.sh [DATA_ROOT] [DATASET] [SHOTS] [BACKBONE]
#
# 示例：
# bash scripts/few_shot_robust.sh data btmri 16 BiomedCLIP
# ========================================

# 参数设置
DATA=$1
DATASET=$2
SHOTS=$3
BACKBONE=$4

# 默认值
if [ -z "$DATA" ]; then
    DATA="data"
fi

if [ -z "$DATASET" ]; then
    DATASET="btmri"
fi

if [ -z "$SHOTS" ]; then
    SHOTS=16
fi

if [ -z "$BACKBONE" ]; then
    BACKBONE="BiomedCLIP"
fi

# 设置 Trainer 名称
case $BACKBONE in
    CLIP)
        TRAINER="BiomedDPT_Robust_CLIP"
        ;;
    BiomedCLIP)
        TRAINER="BiomedDPT_Robust_BiomedCLIP"
        ;;
    PubMedCLIP)
        TRAINER="BiomedDPT_Robust_PubMedCLIP"
        ;;
    PMCCLIP)
        TRAINER="BiomedDPT_Robust_PMCCLIP"
        ;;
    *)
        echo "❌ 未知的 backbone: $BACKBONE"
        exit 1
        ;;
esac

# 配置文件路径
CFG=configs/trainers/BiomedDPT_Robust/few_shot/${DATASET}.yaml

# 输出目录
OUTPUT_DIR=output/BiomedDPT_Robust/${DATASET}_${SHOTS}shots_${BACKBONE}

echo "========================================="
echo "  BiomedDPT_Robust Few-Shot 训练"
echo "========================================="
echo "数据集: $DATASET"
echo "Few-shot: $SHOTS shots"
echo "Backbone: $BACKBONE"
echo "Trainer: $TRAINER"
echo "输出目录: $OUTPUT_DIR"
echo "========================================="
echo ""

# ========== 实验 1：Baseline（不加低质量约束）==========
echo "【Baseline】原始 BiomedDPT（λ3=0）"
python train.py \
    --root $DATA \
    --trainer $TRAINER \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file $CFG \
    --output-dir ${OUTPUT_DIR}_baseline \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.0 \
    DATASET.NUM_SHOTS $SHOTS

echo ""
echo "✅ Baseline 完成"
echo ""

# ========== 实验 2：Robust（低质量约束，模板=minimal）==========
echo "【实验 2】BiomedDPT_Robust（λ3=0.3, 模板=minimal）"
python train.py \
    --root $DATA \
    --trainer $TRAINER \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file $CFG \
    --output-dir ${OUTPUT_DIR}_robust_minimal \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 \
    DATASET.NUM_SHOTS $SHOTS

echo ""
echo "✅ 实验 2 完成"
echo ""

# ========== 实验 3：Robust（低质量约束，模板=article）==========
echo "【实验 3】BiomedDPT_Robust（λ3=0.3, 模板=article）"
python train.py \
    --root $DATA \
    --trainer $TRAINER \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file $CFG \
    --output-dir ${OUTPUT_DIR}_robust_article \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE article \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 \
    DATASET.NUM_SHOTS $SHOTS

echo ""
echo "✅ 实验 3 完成"
echo ""

# ========== 实验 4：Robust（低质量约束，模板=empty）==========
echo "【实验 4】BiomedDPT_Robust（λ3=0.3, 模板=empty）"
python train.py \
    --root $DATA \
    --trainer $TRAINER \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file $CFG \
    --output-dir ${OUTPUT_DIR}_robust_empty \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE empty \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 \
    DATASET.NUM_SHOTS $SHOTS

echo ""
echo "✅ 实验 4 完成"
echo ""

# ========== 实验 5：消融实验（调整 λ3 权重）==========
echo "【实验 5】消融实验（λ3=0.5，加强低质量约束）"
python train.py \
    --root $DATA \
    --trainer $TRAINER \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file $CFG \
    --output-dir ${OUTPUT_DIR}_robust_lambda05 \
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal \
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.5 \
    DATASET.NUM_SHOTS $SHOTS

echo ""
echo "✅ 实验 5 完成"
echo ""

echo "========================================="
echo "  所有实验完成！"
echo "========================================="
echo ""
echo "结果对比（查看各实验目录下的 log.txt）："
echo "1. Baseline:        ${OUTPUT_DIR}_baseline"
echo "2. Robust (minimal): ${OUTPUT_DIR}_robust_minimal"
echo "3. Robust (article): ${OUTPUT_DIR}_robust_article"
echo "4. Robust (empty):   ${OUTPUT_DIR}_robust_empty"
echo "5. Robust (λ3=0.5):  ${OUTPUT_DIR}_robust_lambda05"
echo ""

