#!/bin/bash
# Script to train rank-1 LoRA on all projection matrices for multiple Qwen2.5-Instruct model sizes

echo "Starting rank-1 LoRA training for all projection matrices across model sizes..."
echo "Models: 1.5B, 7B, 14B, 32B"
echo "Rank: 1"
echo "Alpha: 16 (default)"
echo "Learning Rate: 1e-3 (default)"
echo "Dataset: s1K-1.1 (1000 examples)"
echo "Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
echo ""

# Track timing
sweep_start=$(date +%s)
successful_runs=0
failed_runs=0

# Train 1.5B model
echo "========================================="
echo "[1/4] Training Qwen2.5-1.5B-Instruct with rank-1 LoRA..."
echo "========================================="
run_start=$(date +%s)
bash /workspace/s1_peft/train/sft_lora.sh --model_size 1.5B --rank 1
if [ $? -eq 0 ]; then
    successful_runs=$((successful_runs + 1))
    run_end=$(date +%s)
    echo "✓ 1.5B completed in $((run_end - run_start)) seconds"
else
    failed_runs=$((failed_runs + 1))
    echo "✗ 1.5B training failed"
fi

echo ""
echo "========================================="
echo "[2/4] Training Qwen2.5-7B-Instruct with rank-1 LoRA..."
echo "========================================="
run_start=$(date +%s)
bash /workspace/s1_peft/train/sft_lora.sh --model_size 7B --rank 1
if [ $? -eq 0 ]; then
    successful_runs=$((successful_runs + 1))
    run_end=$(date +%s)
    echo "✓ 7B completed in $((run_end - run_start)) seconds"
else
    failed_runs=$((failed_runs + 1))
    echo "✗ 7B training failed"
fi

echo ""
echo "========================================="
echo "[3/4] Training Qwen2.5-14B-Instruct with rank-1 LoRA..."
echo "========================================="
run_start=$(date +%s)
bash /workspace/s1_peft/train/sft_lora.sh --model_size 14B --rank 1
if [ $? -eq 0 ]; then
    successful_runs=$((successful_runs + 1))
    run_end=$(date +%s)
    echo "✓ 14B completed in $((run_end - run_start)) seconds"
else
    failed_runs=$((failed_runs + 1))
    echo "✗ 14B training failed"
fi

echo ""
echo "========================================="
echo "[4/4] Training Qwen2.5-32B-Instruct with rank-1 LoRA..."
echo "========================================="
run_start=$(date +%s)
bash /workspace/s1_peft/train/sft_lora.sh --model_size 32B --rank 1
if [ $? -eq 0 ]; then
    successful_runs=$((successful_runs + 1))
    run_end=$(date +%s)
    echo "✓ 32B completed in $((run_end - run_start)) seconds"
else
    failed_runs=$((failed_runs + 1))
    echo "✗ 32B training failed"
fi

# Calculate total time
sweep_end=$(date +%s)
sweep_duration=$((sweep_end - sweep_start))

echo ""
echo "========================================="
echo "LORA TRAINING SWEEP COMPLETED"
echo "========================================="
echo "Total duration: $((sweep_duration / 3600))h $((sweep_duration % 3600 / 60))m $((sweep_duration % 60))s"
echo "Successful runs: ${successful_runs}/4"
echo "Failed runs: ${failed_runs}/4"
echo ""
echo "Check the following directories for checkpoints:"
echo "  - ckpts_lora/s1-lora-1.5B-r1-*/"
echo "  - ckpts_lora/s1-lora-7B-r1-*/"
echo "  - ckpts_lora/s1-lora-14B-r1-*/"
echo "  - ckpts_lora/s1-lora-32B-r1-*/"
echo ""
echo "To merge and evaluate these models:"
echo "  python merge_lora.py --adapter_path ckpts_lora/s1-lora-<SIZE>-r1-<timestamp>"
echo "  bash eval_lora.sh <merged_model_path>"