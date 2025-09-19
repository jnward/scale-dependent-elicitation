#!/bin/bash

# Comprehensive Evaluation Sweep Script
# Evaluates base models, fine-tuned models, and DeepSeek distill models across all sizes
# Total: 16 evaluations (4 model sizes × 4 variants)

# Default parameters
# TASKS="aime24,aime25"
TASKS="gpqa_diamond_cot_zeroshot,aime24,aime25"
OUTPUT_DIR="results/temperature"  # Default output directory for all evaluation results
GPQA_SAMPLES_FILE="gpqa_samples_120.json"  # Sample indices for GPQA evaluation
SEED=1234  # Default seed for vLLM sampling

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--tasks TASK_LIST] [--output_dir OUTPUT_DIR] [--seed SEED]"
            echo ""
            echo "Options:"
            echo "  --tasks       Comma-separated list of evaluation tasks"
            echo "                Default: gpqa_diamond_cot_zeroshot,aime24,aime25"
            echo "  --output_dir  Output directory for evaluation results"
            echo "                Default: results/temperature"
            echo "  --seed        Random seed for vLLM sampling"
            echo "                Default: 1234"
            echo ""
            echo "Examples:"
            echo "  $0 --tasks aime24,aime25"
            echo "  $0 --output_dir results_run1"
            echo "  $0 --tasks gpqa_diamond_cot_zeroshot --output_dir gpqa_pass1"
            echo "  $0 --seed 42  # Use different seed for varied outputs"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Environment setup
export NCCL_NVLS_ENABLE=0
export NCCL_ALGO=Ring
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

# Fixed parameters
MAX_TOKENS=29000
TENSOR_PARALLEL_SIZE=2
DTYPE="bfloat16"
BATCH_SIZE="auto"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Track sweep start time
sweep_start=$(date +%s)
successful_evals=0
failed_evals=0

echo "========================================="
echo "COMPREHENSIVE EVALUATION SWEEP"
echo "========================================="
echo "Configuration:"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  Data type: ${DTYPE}"
echo "  Tasks: ${TASKS}"
echo "  Output directory: ${OUTPUT_DIR}/"
echo "  Sampling: temperature=0.6, top_p=0.95, seed=${SEED}"
echo ""
echo "Models to evaluate:"
echo "  - Fine-tuned 58-sample models (4 sizes)"
echo "  - Fine-tuned 29-sample models (4 sizes)"
echo "  - Fine-tuned 14-sample models (4 sizes)"
echo "  - Fine-tuned 7-sample models (4 sizes)"
echo "========================================="
echo ""

# Function to run evaluation
run_eval() {
    local model_path=$1
    local output_name=$2
    local model_size=$3
    
    echo "========================================="
    echo "Evaluating: ${output_name}"
    echo "Model path: ${model_path}"
    echo "Output: ${OUTPUT_DIR}/${output_name}"
    echo "========================================="
    
    # Check if we're running GPQA tasks and add samples parameter
    # local SAMPLES_ARG=""
    # if [[ "${TASKS}" == *"gpqa"* ]]; then
    #     if [ -f "${GPQA_SAMPLES_FILE}" ]; then
    #         SAMPLES_ARG="--samples ${GPQA_SAMPLES_FILE}"
    #         echo "Using GPQA samples from: ${GPQA_SAMPLES_FILE}"
    #     else
    #         echo "Warning: GPQA task detected but samples file not found: ${GPQA_SAMPLES_FILE}"
    #     fi
    # fi
    
    # Run evaluation
    lm_eval \
        --model vllm \
        --model_args "pretrained=${model_path},dtype=${DTYPE},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},seed=${SEED}" \
        --tasks ${TASKS} \
        --batch_size ${BATCH_SIZE} \
        --apply_chat_template \
        --output_path "${OUTPUT_DIR}/${output_name}" \
        --log_samples \
        --gen_kwargs "max_gen_toks=${MAX_TOKENS},do_sample=true,temperature=0.6,top_p=0.95"
        # --samples ${GPQA_SAMPLES_FILE}
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        successful_evals=$((successful_evals + 1))
        echo "✓ Completed evaluation for ${output_name}"
    else
        failed_evals=$((failed_evals + 1))
        echo "✗ Failed evaluation for ${output_name}"
    fi
    
    echo ""
    sleep 2  # Brief pause between evaluations
}

# ========================================
# 32B MODELS
# ========================================

echo "===== EVALUATING 32B MODELS ====="
echo ""

# 32B Base Model
run_eval "Qwen/Qwen2.5-32B-Instruct" "32B-base" "32B"

# 32B Fine-tuned 928 samples (using final checkpoint)
run_eval "/workspace/s1_peft/ckpts_sample/32B-928-val72-ep5-20250826_090853_final" "32B-ft-928" "32B"

# 32B Fine-tuned 464 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-464-val72-ep10-20250827_031526" "32B-ft-464" "32B"

# 32B DeepSeek Distill
run_eval "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "32B-deepseek" "32B"

# 32B Fine-tuned 232 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-232-val72-ep20-20250829_224043" "32B-ft-232" "32B"

# 32B Fine-tuned 116 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-116-val72-ep40-20250830_010635" "32B-ft-116" "32B"

# 32B Fine-tuned 58 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-58-val72-ep80-20250903_222637" "32B-ft-58" "32B"

# 32B Fine-tuned 29 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-29-val72-ep160-20250903_235748" "32B-ft-29" "32B"

# 32B Fine-tuned 14 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-14-val72-ep320-20250904_005313" "32B-ft-14" "32B"

# 32B Fine-tuned 7 samples
run_eval "/workspace/s1_peft/ckpts_sample/32B-7-val72-ep640-20250904_013826" "32B-ft-7" "32B"

# 32B LoRA
run_eval "/workspace/s1_peft/ckpts_lora/s1-lora-32B-r1-20250627_013544-merged" "32B-lora" "32B"

# ========================================
# 14B MODELS
# ========================================

echo "===== EVALUATING 14B MODELS ====="
echo ""

# 14B Base Model
run_eval "Qwen/Qwen2.5-14B-Instruct" "14B-base" "14B"

# 14B Fine-tuned 928 samples (using final checkpoint)
run_eval "/workspace/s1_peft/ckpts_sample/14B-928-val72-ep5-20250826_150508_final" "14B-ft-928" "14B"

# 14B Fine-tuned 464 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-464-val72-ep10-20250827_052249" "14B-ft-464" "14B"

# 14B DeepSeek Distill
run_eval "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "14B-deepseek" "14B"

# 14B Fine-tuned 232 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-232-val72-ep20-20250830_000744" "14B-ft-232" "14B"

# 14B Fine-tuned 116 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-116-val72-ep40-20250830_020743" "14B-ft-116" "14B"

# 14B Fine-tuned 58 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-58-val72-ep80-20250903_230810" "14B-ft-58" "14B"

# 14B Fine-tuned 29 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-29-val72-ep160-20250904_003224" "14B-ft-29" "14B"

# 14B Fine-tuned 14 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-14-val72-ep320-20250904_011803" "14B-ft-14" "14B"

# 14B Fine-tuned 7 samples
run_eval "/workspace/s1_peft/ckpts_sample/14B-7-val72-ep640-20250904_020606" "14B-ft-7" "14B"

# 14B LoRA
run_eval "/workspace/s1_peft/ckpts_lora/s1-lora-14B-r1-20250831_200412-merged" "14B-lora" "14B"

# ========================================
# 7B MODELS
# ========================================

echo "===== EVALUATING 7B MODELS ====="
echo ""

# 7B Base Model
run_eval "Qwen/Qwen2.5-7B-Instruct" "7B-base" "7B"

# 7B Fine-tuned 928 samples (using final checkpoint)
run_eval "/workspace/s1_peft/ckpts_sample/7B-928-val72-ep5-20250826_233828_final" "7B-ft-928" "7B"

# 7B Fine-tuned 464 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-464-val72-ep10-20250827_062058" "7B-ft-464" "7B"

# 7B DeepSeek Distill
run_eval "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "7B-deepseek" "7B"

# 7B Fine-tuned 232 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-232-val72-ep20-20250830_004433" "7B-ft-232" "7B"

# 7B Fine-tuned 116 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-116-val72-ep40-20250830_023324" "7B-ft-116" "7B"

# 7B Fine-tuned 58 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-58-val72-ep80-20250903_234631" "7B-ft-58" "7B"

# 7B Fine-tuned 29 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-29-val72-ep160-20250904_004424" "7B-ft-29" "7B"

# 7B Fine-tuned 14 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-14-val72-ep320-20250904_013022" "7B-ft-14" "7B"

# 7B Fine-tuned 7 samples
run_eval "/workspace/s1_peft/ckpts_sample/7B-7-val72-ep640-20250904_021546" "7B-ft-7" "7B"

# 7B LoRA (commented out for now)
run_eval "/workspace/s1_peft/ckpts_lora/s1-lora-7B-r1-20250831_192937-merged" "7B-lora" "7B"

# ========================================
# 1.5B MODELS
# ========================================

echo "===== EVALUATING 1.5B MODELS ====="
echo ""

# 1.5B Base Model
run_eval "Qwen/Qwen2.5-1.5B-Instruct" "1.5B-base" "1.5B"

# 1.5B Fine-tuned 928 samples (using final checkpoint)
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-928-val72-ep5-20250827_002441_final" "1.5B-ft-928" "1.5B"

# 1.5B Fine-tuned 464 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-464-val72-ep10-20250827_064401" "1.5B-ft-464" "1.5B"

# 1.5B DeepSeek Distill
run_eval "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "1.5B-deepseek" "1.5B"

# 1.5B Fine-tuned 232 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-232-val72-ep20-20250830_030812" "1.5B-ft-232" "1.5B"

# 1.5B Fine-tuned 116 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-116-val72-ep40-20250830_031425" "1.5B-ft-116" "1.5B"

# 1.5B Fine-tuned 58 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-58-val72-ep80-20250903_235504" "1.5B-ft-58" "1.5B"

# 1.5B Fine-tuned 29 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-29-val72-ep160-20250904_005027" "1.5B-ft-29" "1.5B"

# 1.5B Fine-tuned 14 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-14-val72-ep320-20250904_013621" "1.5B-ft-14" "1.5B"

# 1.5B Fine-tuned 7 samples
run_eval "/workspace/s1_peft/ckpts_sample/1.5B-7-val72-ep640-20250904_022038" "1.5B-ft-7" "1.5B"

# 1.5B LoRA (commented out for now)
run_eval "/workspace/s1_peft/ckpts_lora/s1-lora-1.5B-r1-20250831_190657-merged" "1.5B-lora" "1.5B"

# ========================================
# SUMMARY
# ========================================

# Calculate total sweep time
sweep_end=$(date +%s)
sweep_duration=$((sweep_end - sweep_start))

echo "========================================="
echo "EVALUATION SWEEP COMPLETED"
echo "========================================="
echo "Total duration: $((sweep_duration / 3600))h $((sweep_duration % 3600 / 60))m $((sweep_duration % 60))s"
echo "Successful evaluations: ${successful_evals}/16"
echo "Failed evaluations: ${failed_evals}/16"
echo ""

echo "Results saved in ${OUTPUT_DIR}/:"
echo ""
ls -la ${OUTPUT_DIR}/

echo ""
echo "To view results for a specific model:"
echo "  cat ${OUTPUT_DIR}/MODEL_NAME/results.json"
echo ""
echo "To compare scores across models:"
echo "  for dir in ${OUTPUT_DIR}/*/; do"
echo "    echo \"\$(basename \$dir):\""
echo "    grep -h \"acc\" \"\$dir/results.json\" 2>/dev/null | head -5"
echo "  done"
echo ""

# Create a summary file
echo "Creating summary file..."
{
    echo "Evaluation Sweep Summary"
    echo "========================"
    echo "Date: $(date)"
    echo "Total duration: $((sweep_duration / 3600))h $((sweep_duration % 3600 / 60))m"
    echo "Successful: ${successful_evals}/16"
    echo "Failed: ${failed_evals}/16"
    echo ""
    echo "Models evaluated:"
    for dir in ${OUTPUT_DIR}/*/; do
        if [ -d "$dir" ]; then
            echo "  - $(basename $dir)"
        fi
    done
} > ${OUTPUT_DIR}/evaluation_summary.txt

echo "Summary saved to ${OUTPUT_DIR}/evaluation_summary.txt"
echo ""
echo "Sweep complete!"