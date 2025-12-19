# ========================================
# BiomedDPT_Robust Few-Shot 训练脚本（PowerShell 版本）
# 文件位置：scripts/few_shot_robust.ps1
#
# 用法：
# .\scripts\few_shot_robust.ps1 -DATA "data" -DATASET "btmri" -SHOTS 16 -BACKBONE "BiomedCLIP"
# ========================================

param(
    [string]$DATA = "data",
    [string]$DATASET = "btmri",
    [int]$SHOTS = 16,
    [string]$BACKBONE = "BiomedCLIP"
)

# 设置 Trainer 名称
$TRAINER = switch ($BACKBONE) {
    "CLIP"        { "BiomedDPT_Robust_CLIP" }
    "BiomedCLIP"  { "BiomedDPT_Robust_BiomedCLIP" }
    "PubMedCLIP"  { "BiomedDPT_Robust_PubMedCLIP" }
    "PMCCLIP"     { "BiomedDPT_Robust_PMCCLIP" }
    default {
        Write-Host "❌ 未知的 backbone: $BACKBONE" -ForegroundColor Red
        exit 1
    }
}

# 配置文件路径
$CFG = "configs/trainers/BiomedDPT_Robust/few_shot/$DATASET.yaml"
$DATASET_CFG = "configs/datasets/$DATASET.yaml"

# 输出目录
$OUTPUT_BASE = "output/BiomedDPT_Robust/${DATASET}_${SHOTS}shots_${BACKBONE}"

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  BiomedDPT_Robust Few-Shot 训练" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "数据集: $DATASET" -ForegroundColor Yellow
Write-Host "Few-shot: $SHOTS shots" -ForegroundColor Yellow
Write-Host "Backbone: $BACKBONE" -ForegroundColor Yellow
Write-Host "Trainer: $TRAINER" -ForegroundColor Yellow
Write-Host "输出目录: $OUTPUT_BASE" -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# 设置 CUDA 设备
$env:CUDA_VISIBLE_DEVICES = "0"

# ========== 实验 1：Baseline（不加低质量约束）==========
Write-Host "【Baseline】原始 BiomedDPT（λ3=0）" -ForegroundColor Green
python train.py `
    --root $DATA `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CFG `
    --config-file $CFG `
    --output-dir "${OUTPUT_BASE}_baseline" `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.0 `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ Baseline 完成`n" -ForegroundColor Green

# ========== 实验 2：Robust（低质量约束，模板=minimal）==========
Write-Host "【实验 2】BiomedDPT_Robust（λ3=0.3, 模板=minimal）" -ForegroundColor Green
python train.py `
    --root $DATA `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CFG `
    --config-file $CFG `
    --output-dir "${OUTPUT_BASE}_robust_minimal" `
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 2 完成`n" -ForegroundColor Green

# ========== 实验 3：Robust（低质量约束，模板=article）==========
Write-Host "【实验 3】BiomedDPT_Robust（λ3=0.3, 模板=article）" -ForegroundColor Green
python train.py `
    --root $DATA `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CFG `
    --config-file $CFG `
    --output-dir "${OUTPUT_BASE}_robust_article" `
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE article `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 3 完成`n" -ForegroundColor Green

# ========== 实验 4：Robust（低质量约束，模板=empty）==========
Write-Host "【实验 4】BiomedDPT_Robust（λ3=0.3, 模板=empty）" -ForegroundColor Green
python train.py `
    --root $DATA `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CFG `
    --config-file $CFG `
    --output-dir "${OUTPUT_BASE}_robust_empty" `
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE empty `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 4 完成`n" -ForegroundColor Green

# ========== 实验 5：消融实验（调整 λ3 权重）==========
Write-Host "【实验 5】消融实验（λ3=0.5，加强低质量约束）" -ForegroundColor Green
python train.py `
    --root $DATA `
    --trainer $TRAINER `
    --dataset-config-file $DATASET_CFG `
    --config-file $CFG `
    --output-dir "${OUTPUT_BASE}_robust_lambda05" `
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.5 `
    DATASET.NUM_SHOTS $SHOTS

Write-Host "`n✅ 实验 5 完成`n" -ForegroundColor Green

Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "  所有实验完成！" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "`n结果对比（查看各实验目录下的 log.txt）：" -ForegroundColor Yellow
Write-Host "1. Baseline:         ${OUTPUT_BASE}_baseline" -ForegroundColor White
Write-Host "2. Robust (minimal): ${OUTPUT_BASE}_robust_minimal" -ForegroundColor White
Write-Host "3. Robust (article): ${OUTPUT_BASE}_robust_article" -ForegroundColor White
Write-Host "4. Robust (empty):   ${OUTPUT_BASE}_robust_empty" -ForegroundColor White
Write-Host "5. Robust (λ3=0.5):  ${OUTPUT_BASE}_robust_lambda05`n" -ForegroundColor White
