# ========================================
# 快速测试脚本（用于验证代码是否正常运行）
# 文件位置：scripts/quick_test_robust.ps1
#
# 功能：运行 5 个 epoch 快速测试
# ========================================

$env:CUDA_VISIBLE_DEVICES = "0"

Write-Host "`n🧪 快速测试 BiomedDPT_Robust（5 epochs）`n" -ForegroundColor Cyan

python train.py `
    --root data `
    --trainer BiomedDPT_Robust_BiomedCLIP `
    --dataset-config-file configs/datasets/btmri.yaml `
    --config-file configs/trainers/BiomedDPT_Robust/few_shot/btmri.yaml `
    --output-dir output/quick_test_robust `
    TRAINER.BIOMEDDPT_ROBUST.LOW_TEMPLATE_TYPE minimal `
    TRAINER.BIOMEDDPT_ROBUST.L1_LAMBDA_LOW 0.3 `
    DATASET.NUM_SHOTS 16 `
    OPTIM.MAX_EPOCH 5

Write-Host "`n✅ 快速测试完成！`n" -ForegroundColor Green
