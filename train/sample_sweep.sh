#!/bin/bash

# Parse command line arguments
MODEL_SIZE="1.5B"

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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--model-size SIZE]"
            echo "  --model-size: Model size: 1.5B, 7B, 14B, or 32B (default: 1.5B)"
            exit 1
            ;;
    esac
done

# Configuration for sweep
# Each configuration maintains constant total training steps (1000 examples * 5 epochs = 5000 steps)
declare -a SAMPLES=(125 250 500 1000)
declare -a EPOCHS=(40 20 10 5)
RANDOM_SEED=42

echo "========================================="
echo "Sample Size Sweep Configuration"
echo "========================================="
echo "Model size: ${MODEL_SIZE}"
echo "Random seed: ${RANDOM_SEED}"
echo "Configurations:"
for i in ${!SAMPLES[@]}; do
    echo "  - ${SAMPLES[$i]} samples × ${EPOCHS[$i]} epochs = $((${SAMPLES[$i]} * ${EPOCHS[$i]})) total steps"
done
echo "========================================="
echo ""

# Create output directory if it doesn't exist
mkdir -p ckpts_sample

# Run training for each configuration
for i in ${!SAMPLES[@]}; do
    n_samples=${SAMPLES[$i]}
    n_epochs=${EPOCHS[$i]}
    
    echo "========================================="
    echo "Starting training run $((i+1)) of ${#SAMPLES[@]}"
    echo "Samples: ${n_samples}, Epochs: ${n_epochs}"
    echo "========================================="
    
    # Run the training script
    bash /workspace/s1_peft/train/sft_sample.sh \
        --model-size "${MODEL_SIZE}" \
        --n-examples ${n_samples} \
        --n-epochs ${n_epochs} \
        --random-seed ${RANDOM_SEED}
    
    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "Error: Training failed for ${n_samples} samples"
        exit 1
    fi
    
    echo ""
    echo "Completed training for ${n_samples} samples"
    echo ""
    
    # Optional: Add a small delay between runs to ensure clean separation
    sleep 2
done

echo "========================================="
echo "Sample sweep completed successfully!"
echo "========================================="
echo "Results saved in ckpts_sample/"
echo ""
echo "To evaluate the models, you can use:"
echo "  ls -la ckpts_sample/${MODEL_SIZE}-*"