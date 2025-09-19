#!/bin/bash

# Full Model + Dataset Size Sweep Script
# Sweeps across both model sizes (32B, 14B, 7B, 1.5B) and dataset sizes (928, 464, 232, 116)
# Each run can be individually disabled by commenting out its line in the RUNS array

# Parse command line arguments
EVAL_STEPS=16
EARLY_STOPPING_PATIENCE=2
LOSS_EPSILON=0.1
DRY_RUN=false
SKIP_COMPLETED=false
RESUME_FROM=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --loss-epsilon)
            LOSS_EPSILON="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--eval-steps N] [--early-stopping-patience N] [--loss-epsilon E] [--dry-run] [--skip-completed] [--resume-from N]"
            echo "  --eval-steps: Evaluate every N optimizer steps (default: 16)"
            echo "  --early-stopping-patience: Stop if no improvement for N evaluations (default: 5)"
            echo "  --loss-epsilon: Tolerance for validation loss increase in nats (default: 0.06)"
            echo "  --dry-run: Show what would be executed without running"
            echo "  --skip-completed: Skip runs that already have checkpoints"
            echo "  --resume-from: Start from run number N (default: 1)"
            exit 1
            ;;
    esac
done

# ========================================
# RUN CONFIGURATIONS
# ========================================
# Format: "MODEL_SIZE N_SAMPLES N_EPOCHS"
# Comment out any line to skip that specific run
# Ordered by: all model sizes for each dataset size (928 first, then 464, etc.)

RUNS=(
    # ===== 928 samples (full dataset) =====
    "32B 928 5"     # 32B with full dataset
    "14B 928 5"     # 14B with full dataset
    "7B 928 5"      # 7B with full dataset
    "1.5B 928 5"    # 1.5B with full dataset
    
    # ===== 464 samples =====
    "32B 464 10"    # 32B with 464 samples
    "14B 464 10"    # 14B with 464 samples
    "7B 464 10"     # 7B with 464 samples
    "1.5B 464 10"   # 1.5B with 464 samples
    
    # ===== 232 samples =====
    "32B 232 20"    # 32B with 232 samples
    "14B 232 20"    # 14B with 232 samples
    "7B 232 20"     # 7B with 232 samples
    "1.5B 232 20"   # 1.5B with 232 samples
    
    # ===== 116 samples =====
    "32B 116 40"    # 32B with 116 samples
    "14B 116 40"    # 14B with 116 samples
    "7B 116 40"     # 7B with 116 samples
    "1.5B 116 40"   # 1.5B with 116 samples
    
    # ===== 58 samples =====
    "32B 58 80"     # 32B with 58 samples
    "14B 58 80"     # 14B with 58 samples
    "7B 58 80"      # 7B with 58 samples
    "1.5B 58 80"    # 1.5B with 58 samples
    
    # ===== 29 samples =====
    "32B 29 160"    # 32B with 29 samples
    "14B 29 160"    # 14B with 29 samples
    "7B 29 160"     # 7B with 29 samples
    "1.5B 29 160"   # 1.5B with 29 samples
    
    # ===== 14 samples =====
    "32B 14 320"    # 32B with 14 samples
    "14B 14 320"    # 14B with 14 samples
    "7B 14 320"     # 7B with 14 samples
    "1.5B 14 320"   # 1.5B with 14 samples
    
    # ===== 7 samples =====
    "32B 7 640"     # 32B with 7 samples
    "14B 7 640"     # 14B with 7 samples
    "7B 7 640"      # 7B with 7 samples
    "1.5B 7 640"    # 1.5B with 7 samples
)

# Fixed parameters for all runs
RANDOM_SEED=42
N_VAL_EXAMPLES=72

# ========================================
# DISPLAY CONFIGURATION
# ========================================

echo "========================================="
echo "Full Model + Dataset Size Sweep"
echo "========================================="
echo "Configuration:"
echo "  Random seed: ${RANDOM_SEED}"
echo "  Validation set: ${N_VAL_EXAMPLES} examples (fixed)"
echo "  Evaluation frequency: Every ${EVAL_STEPS} optimizer steps"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Loss epsilon: ${LOSS_EPSILON}"
echo "  Dry run: ${DRY_RUN}"
echo "  Skip completed: ${SKIP_COMPLETED}"
echo "  Resume from run: ${RESUME_FROM}"
echo ""

# Count and display planned runs
total_runs=${#RUNS[@]}
echo "Planned runs (${total_runs} total):"
echo ""

run_num=0
for config in "${RUNS[@]}"; do
    run_num=$((run_num + 1))
    # Skip if before resume point
    if [ $run_num -lt $RESUME_FROM ]; then
        continue
    fi
    
    # Parse configuration
    read -r model_size n_samples n_epochs <<< "$config"
    total_steps=$((n_samples * n_epochs))
    
    # Format run number with padding for alignment
    printf "  Run %2d: %-4s model, %3d samples × %2d epochs = %4d steps" \
           $run_num $model_size $n_samples $n_epochs $total_steps
    
    # Check if checkpoint already exists (if skip-completed is enabled)
    if [ "$SKIP_COMPLETED" = true ]; then
        existing=$(ls -d ckpts_sample/${model_size}-${n_samples}-val${N_VAL_EXAMPLES}-* 2>/dev/null | head -1)
        if [ -n "$existing" ]; then
            echo " [SKIP - exists: $(basename $existing)]"
        else
            echo ""
        fi
    else
        echo ""
    fi
done

echo ""
echo "========================================="

# Exit if dry run
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - Exiting without executing"
    exit 0
fi

# ========================================
# EXECUTION
# ========================================

# Create output directory if it doesn't exist
mkdir -p ckpts_sample

# Track sweep start time
sweep_start=$(date +%s)
successful_runs=0
failed_runs=0
skipped_runs=0

# Arrays to track results for summary
declare -a completed_configs=()
declare -a completed_times=()
declare -a completed_paths=()

# Run training for each configuration
run_num=0
for config in "${RUNS[@]}"; do
    run_num=$((run_num + 1))
    
    # Skip if before resume point
    if [ $run_num -lt $RESUME_FROM ]; then
        echo "Skipping run $run_num (before resume point)"
        skipped_runs=$((skipped_runs + 1))
        continue
    fi
    
    # Parse configuration
    read -r model_size n_samples n_epochs <<< "$config"
    
    # Check if should skip completed
    if [ "$SKIP_COMPLETED" = true ]; then
        existing=$(ls -d ckpts_sample/${model_size}-${n_samples}-val${N_VAL_EXAMPLES}-* 2>/dev/null | head -1)
        if [ -n "$existing" ]; then
            echo "========================================="
            echo "Skipping run $run_num of ${total_runs} (already exists)"
            echo "Model: ${model_size}, Samples: ${n_samples}"
            echo "Existing checkpoint: $(basename $existing)"
            echo "========================================="
            skipped_runs=$((skipped_runs + 1))
            continue
        fi
    fi
    
    echo "========================================="
    echo "Starting run $run_num of ${total_runs}"
    echo "Model: ${model_size}, Train samples: ${n_samples}, Epochs: ${n_epochs}"
    echo "Validation samples: ${N_VAL_EXAMPLES}"
    
    # Special handling for full dataset
    if [ ${n_samples} -eq 928 ]; then
        echo "Full dataset: Will save both best and final checkpoints (no early stopping)"
    fi
    
    # Show progress
    completed=$((run_num - RESUME_FROM - skipped_runs))
    remaining=$((total_runs - run_num))
    if [ $completed -gt 0 ] && [ ${#completed_times[@]} -gt 0 ]; then
        avg_time=0
        for t in "${completed_times[@]}"; do
            avg_time=$((avg_time + t))
        done
        avg_time=$((avg_time / ${#completed_times[@]}))
        est_remaining=$((avg_time * remaining))
        echo "Progress: $completed completed, $remaining remaining"
        echo "Estimated time remaining: $((est_remaining / 3600))h $((est_remaining % 3600 / 60))m"
    fi
    echo "========================================="
    
    # Create custom output directory with timestamp
    uid="$(date +%Y%m%d_%H%M%S)"
    export CUSTOM_OUTPUT_DIR="ckpts_sample/${model_size}-${n_samples}-val${N_VAL_EXAMPLES}-ep${n_epochs}-${uid}"
    
    # Run start time
    run_start=$(date +%s)
    
    # Use same eval steps for all models to avoid wandb step counting issues
    eval_steps_adjusted=$EVAL_STEPS
    
    # Run the training script
    if [ ${n_samples} -eq 928 ]; then
        # Full dataset: save final checkpoint
        bash /workspace/s1_peft/train/sft_sample_val.sh \
            --model-size "${model_size}" \
            --n-examples ${n_samples} \
            --n-epochs ${n_epochs} \
            --random-seed ${RANDOM_SEED} \
            --n-val-examples ${N_VAL_EXAMPLES} \
            --eval-steps ${eval_steps_adjusted} \
            --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
            --loss-epsilon ${LOSS_EPSILON} \
            --save-final-checkpoint
    else
        bash /workspace/s1_peft/train/sft_sample_val.sh \
            --model-size "${model_size}" \
            --n-examples ${n_samples} \
            --n-epochs ${n_epochs} \
            --random-seed ${RANDOM_SEED} \
            --n-val-examples ${N_VAL_EXAMPLES} \
            --eval-steps ${eval_steps_adjusted} \
            --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
            --loss-epsilon ${LOSS_EPSILON}
    fi
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        successful_runs=$((successful_runs + 1))
        
        # Calculate run time
        run_end=$(date +%s)
        run_duration=$((run_end - run_start))
        
        # Store results
        completed_configs+=("${model_size}-${n_samples}")
        completed_times+=($run_duration)
        completed_paths+=("${CUSTOM_OUTPUT_DIR}")
        
        echo ""
        echo "✓ Completed ${model_size} with ${n_samples} samples in ${run_duration} seconds"
        echo "  Checkpoint saved to: ${CUSTOM_OUTPUT_DIR}"
        
        # Display metrics if available
        if [ -f "${CUSTOM_OUTPUT_DIR}/final_metrics.json" ]; then
            best_loss=$(grep -o '"best_eval_loss": [0-9.]*' "${CUSTOM_OUTPUT_DIR}/final_metrics.json" | cut -d' ' -f2)
            if [ -n "$best_loss" ]; then
                echo "  Best validation loss: ${best_loss}"
            fi
        fi
    else
        failed_runs=$((failed_runs + 1))
        echo ""
        echo "✗ ERROR: Training failed for ${model_size} with ${n_samples} samples"
        
        # Optionally continue or exit on failure
        read -p "Continue with remaining runs? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping sweep due to failure"
            break
        fi
    fi
    
    echo ""
    
    # Optional: Add a small delay between runs
    sleep 2
done

# ========================================
# SUMMARY
# ========================================

# Calculate total sweep time
sweep_end=$(date +%s)
sweep_duration=$((sweep_end - sweep_start))

echo "========================================="
echo "SWEEP COMPLETED"
echo "========================================="
echo "Total duration: $((sweep_duration / 3600))h $((sweep_duration % 3600 / 60))m $((sweep_duration % 60))s"
echo "Successful runs: ${successful_runs}"
echo "Failed runs: ${failed_runs}"
echo "Skipped runs: ${skipped_runs}"
echo ""

if [ ${#completed_paths[@]} -gt 0 ]; then
    echo "Completed checkpoints:"
    for i in "${!completed_configs[@]}"; do
        duration=${completed_times[$i]}
        echo "  ${completed_configs[$i]}: ${duration}s - ${completed_paths[$i]}"
    done
    echo ""
fi

echo "To view all results:"
echo "  ls -la ckpts_sample/*-val${N_VAL_EXAMPLES}-*"
echo ""

echo "To compare validation losses by model size:"
for model in 32B 14B 7B 1.5B; do
    echo "  ${model}:"
    for samples in 928 464 232 116 58 29 14 7; do
        result=$(grep -h best_eval_loss ckpts_sample/${model}-${samples}-val${N_VAL_EXAMPLES}-*/final_metrics*.json 2>/dev/null | head -1)
        if [ -n "$result" ]; then
            loss=$(echo $result | grep -o '[0-9.]*' | head -1)
            printf "    n=%3d: loss=%.4f\n" $samples $loss
        else
            printf "    n=%3d: no checkpoint\n" $samples
        fi
    done
done

echo ""
echo "Sweep complete!"