#!/bin/bash
# Orchestrator script for analysis pipeline

set -e  # Exit on error

echo "============================================"
echo "Starting Analysis Pipeline"
echo "============================================"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run performance recovery plots
echo ""
echo "============================================"
echo "Creating Performance Recovery Plots"
echo "============================================"
python plot_performance_recovery.py

# Run LoRA performance plots
echo ""
echo "============================================"
echo "Creating LoRA Performance Plots"
echo "============================================"
python plot_lora_performance.py

# Run validation loss plots
echo ""
echo "============================================"
echo "Creating Validation Loss Plots"
echo "============================================"
python plot_validation_loss.py

echo ""
echo "============================================"
echo "Analysis Complete!"
echo "============================================"
echo ""
echo "All plots have been saved to the plots/ directory"
echo "CSV data has been saved to performance_recovered.csv"
echo "Summary tables have been printed above"