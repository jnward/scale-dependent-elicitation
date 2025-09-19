#!/bin/bash

# Parse command line arguments
MODEL_SIZE="1.5B"
EVAL_STEPS=16
EARLY_STOPPING_PATIENCE=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-size)
            MODEL_SIZE="$2"
            if [[ ! "$MODEL_SIZE" =~ ^(1\.5B|7B|14B|32B)$ ]]; then
                echo "Error: Invalid model size. Must be one of: 1.5B, 7B, 14B, 32B"
                exit 1
            fi
            shift 2
            ;;
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--model-size SIZE] [--eval-steps N] [--early-stopping-patience N]"
            echo "  --model-size: Model size: 1.5B, 7B, 14B, or 32B (default: 1.5B)"
            echo "  --eval-steps: Evaluate every N optimizer steps (default: 16)"
            echo "  --early-stopping-patience: Stop if no improvement for N evaluations (default: 5, 0 to disable)"
            exit 1
            ;;
    esac
done

# Configuration for sweep
# Hierarchical dataset sizes: 7 ⊂ 14 ⊂ 29 ⊂ 58 ⊂ 116 ⊂ 232 ⊂ 464 ⊂ 928
# Each maintains constant total training steps for comparison
# Run largest first (longest training, good for early debugging)
# Original runs (already completed):
# declare -a SAMPLES=(928 464 232 116)
# declare -a EPOCHS=(5 10 20 40)  # Adjusted to maintain ~5000 total steps
# Extended runs with smaller sample sizes:
declare -a SAMPLES=(58 29 14 7)
declare -a EPOCHS=(80 160 320 640)  # Adjusted to maintain ~4640 total steps
RANDOM_SEED=42
N_VAL_EXAMPLES=72

echo "========================================="
echo "Validation Sample Size Sweep Configuration"
echo "========================================="
echo "Model size: ${MODEL_SIZE}"
echo "Random seed: ${RANDOM_SEED}"
echo "Validation set: ${N_VAL_EXAMPLES} examples (fixed for all runs)"
echo "Evaluation frequency: Every ${EVAL_STEPS} optimizer steps"
echo "Early stopping: After ${EARLY_STOPPING_PATIENCE} evaluations without improvement"
echo ""
echo "Training configurations (largest to smallest):"
for i in ${!SAMPLES[@]}; do
    total_steps=$((${SAMPLES[$i]} * ${EPOCHS[$i]}))
    echo "  Run $((i+1)): ${SAMPLES[$i]} train × ${EPOCHS[$i]} epochs = ${total_steps} total steps"
done
echo ""
echo "Dataset hierarchy: 116 ⊂ 232 ⊂ 464 ⊂ 928 (all from same pool)"
echo "Execution order: 928 → 464 → 232 → 116 (largest first)"
echo "Validation set: Same 72 examples for all runs"
echo ""
echo "Checkpoint strategy: Save to local /tmp during training,"
echo "                     copy to network only at end (much faster)"
echo "To disable local storage: export DISABLE_LOCAL_CHECKPOINT=true"
echo "========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p ckpts_sample

# Track sweep start time
sweep_start=$(date +%s)

# Run training for each configuration
for i in ${!SAMPLES[@]}; do
    n_samples=${SAMPLES[$i]}
    n_epochs=${EPOCHS[$i]}
    
    echo "========================================="
    echo "Starting training run $((i+1)) of ${#SAMPLES[@]}"
    echo "Train samples: ${n_samples}, Epochs: ${n_epochs}"
    echo "Validation samples: ${N_VAL_EXAMPLES}"
    if [ ${n_samples} -eq 928 ]; then
        echo "Full dataset: Will save both best and final checkpoints (no early stopping)"
    fi
    echo "========================================="
    
    # Create custom output directory with validation info
    uid="$(date +%Y%m%d_%H%M%S)"
    export CUSTOM_OUTPUT_DIR="ckpts_sample/${MODEL_SIZE}-${n_samples}-val${N_VAL_EXAMPLES}-ep${n_epochs}-${uid}"
    
    # Run start time
    run_start=$(date +%s)
    
    # Run the training script
    # For 928 samples (full dataset), add --save-final-checkpoint flag
    if [ ${n_samples} -eq 928 ]; then
        bash /workspace/s1_peft/train/sft_sample_val.sh \
            --model-size "${MODEL_SIZE}" \
            --n-examples ${n_samples} \
            --n-epochs ${n_epochs} \
            --random-seed ${RANDOM_SEED} \
            --n-val-examples ${N_VAL_EXAMPLES} \
            --eval-steps ${EVAL_STEPS} \
            --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
            --save-final-checkpoint
    else
        bash /workspace/s1_peft/train/sft_sample_val.sh \
            --model-size "${MODEL_SIZE}" \
            --n-examples ${n_samples} \
            --n-epochs ${n_epochs} \
            --random-seed ${RANDOM_SEED} \
            --n-val-examples ${N_VAL_EXAMPLES} \
            --eval-steps ${EVAL_STEPS} \
            --early-stopping-patience ${EARLY_STOPPING_PATIENCE}
    fi
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for ${n_samples} samples"
        exit 1
    fi
    
    # Calculate run time
    run_end=$(date +%s)
    run_duration=$((run_end - run_start))
    
    echo ""
    echo "Completed training for ${n_samples} samples in ${run_duration} seconds"
    echo "Best model saved to: ${CUSTOM_OUTPUT_DIR}"
    
    # Display metrics if available
    if [ -f "${CUSTOM_OUTPUT_DIR}/final_metrics.json" ]; then
        echo "Final metrics:"
        cat "${CUSTOM_OUTPUT_DIR}/final_metrics.json" | head -20
    fi
    
    echo ""
    
    # Optional: Add a small delay between runs to ensure clean separation
    sleep 2
done

# Calculate total sweep time
sweep_end=$(date +%s)
sweep_duration=$((sweep_end - sweep_start))

echo "========================================="
echo "Validation sample sweep completed successfully!"
echo "========================================="
echo "Total sweep duration: ${sweep_duration} seconds"
echo ""
echo "Results saved in ckpts_sample/"
echo ""
echo "To compare validation losses across runs (in size order):"
echo "  for samples in 928 464 232 116; do"
echo "    echo \"n=\$samples:\""
echo "    grep best_eval_loss ckpts_sample/${MODEL_SIZE}-\${samples}-val${N_VAL_EXAMPLES}-*/final_metrics*.json 2>/dev/null || echo \"  No checkpoint saved\""
echo "  done"
echo ""
echo "To list all checkpoints:"
echo "  ls -la ckpts_sample/${MODEL_SIZE}-*-val${N_VAL_EXAMPLES}-*"
echo ""
echo "Dataset indices are saved in each checkpoint directory:"
echo "  - dataset_indices.json: Contains exact train/val split indices"
echo "  - best_model_info.json: Contains best validation loss and step"
echo "  - final_metrics.json: Contains all final training metrics"