#!/usr/bin/env python3
"""
Plot LoRA performance comparison graphs
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from data_utils import (
    combine_aime_scores,
    get_variant_label,
    parse_gpqa_multiple_runs,
    PLOTS_DIR
)


def create_lora_comparison_bar_plot(results: Dict):
    """Create horizontal bar charts showing absolute performance for LoRA comparison"""

    # Combine AIME scores first
    combined_results = combine_aime_scores(results)

    # Models to compare (in order for the plot - top to bottom)
    models = ['lora', 'ft-58', 'ft-116', 'ft-928', 'deepseek', 'base']
    model_labels = ['LoRA (full dataset)', 'FT 58 samples', 'FT 116 samples', 'FT 928 samples', 'DeepSeek-R1-Distill-Qwen-32B', 'Qwen2.5-32B-Instruct']

    # Colors for each model
    colors = {
        'base': '#808080',      # Gray
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',    # Light red
        'ft-928': '#A23B72',    # Purple
        'deepseek': '#2A9D8F',  # Teal
        'lora': '#FFA500'       # Orange
    }

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Collect AIME scores
    aime_scores = []
    aime_errors = []

    for model in models:
        if model == 'lora':
            # Get LoRA AIME scores
            if '32B' in combined_results and 'lora' in combined_results['32B']:
                score = combined_results['32B']['lora'].get('aime_combined', {}).get('score', 0) * 100
                stderr = combined_results['32B']['lora'].get('aime_combined', {}).get('stderr', 0) * 100
            else:
                score = 0
                stderr = 0
        else:
            # Get regular model scores
            score = combined_results.get('32B', {}).get(model, {}).get('aime_combined', {}).get('score', 0) * 100
            stderr = combined_results.get('32B', {}).get(model, {}).get('aime_combined', {}).get('stderr', 0) * 100

        aime_scores.append(score)
        aime_errors.append(stderr)

    # Collect GPQA scores (averaging across multiple runs)
    gpqa_scores = []
    gpqa_errors = []

    # Get GPQA scores for LoRA from multiple runs
    def get_gpqa_scores_with_lora(model_key):
        if model_key == 'lora':
            # Collect LoRA scores from all GPQA runs
            scores = []
            dirs = [
                "results/temperature",
                "results/temperature_gpqa_2",
                "results/temperature_gpqa_3",
                "results/temperature_gpqa_4"
            ]

            for dir_name in dirs:
                dir_path = Path(dir_name) / "32B-lora"
                if dir_path.exists():
                    for subdir in dir_path.iterdir():
                        if subdir.is_dir():
                            json_files = list(subdir.glob("results_*.json"))
                            if json_files:
                                json_file = sorted(json_files)[-1]
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                    if "results" in data and "gpqa_diamond_cot_zeroshot" in data["results"]:
                                        score = data["results"]["gpqa_diamond_cot_zeroshot"].get("exact_match,none", None)
                                        if score is not None:
                                            scores.append(score)
                                            break
                                except Exception as e:
                                    print(f"Warning: Could not read {json_file}: {e}")

            if scores:
                mean_score = np.mean(scores) * 100
                if len(scores) > 1:
                    std_score = np.std(scores, ddof=1) * 100
                    stderr = std_score / np.sqrt(len(scores))
                else:
                    stderr = 0
                return mean_score, stderr
            return 0, 0
        else:
            # Use existing parse_gpqa_multiple_runs function for other models
            mean, stderr = parse_gpqa_multiple_runs('32B', model_key)
            return mean * 100, stderr * 100

    for model in models:
        score, stderr = get_gpqa_scores_with_lora(model)
        gpqa_scores.append(score)
        gpqa_errors.append(stderr)

    # Plot 1: AIME (left)
    ax = axes[0]
    y_positions = np.arange(len(models))[::-1]  # Reverse order for top-to-bottom display

    # Create horizontal bars
    bars = ax.barh(y_positions, aime_scores, xerr=aime_errors,
                   color=[colors[m] for m in models],
                   edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel('Score (%)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, aime_scores)):
        if score > 0:
            ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}%', va='center', fontsize=9)

    # Plot 2: GPQA (right)
    ax = axes[1]

    # Use same y_positions as AIME plot for consistency
    # Create horizontal bars
    bars = ax.barh(y_positions, gpqa_scores, xerr=gpqa_errors,
                   color=[colors[m] for m in models],
                   edgecolor='black', linewidth=0.5, capsize=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_labels)
    ax.set_xlabel('Score (%)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, gpqa_scores)):
        if score > 0:
            ax.text(score + 1, bar.get_y() + bar.get_height()/2,
                   f'{score:.1f}%', va='center', fontsize=9)

    # Overall title
    # plt.suptitle('32B Model Performance Comparison: LoRA vs Fine-tuning',
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    plot_path = PLOTS_DIR / 'lora_comparison_bars.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def create_lora_comparison_sidebyside_bar_plot(results: Dict):
    """Create side-by-side vertical bar charts showing absolute performance for LoRA comparison across all model sizes"""

    # Combine AIME scores first
    combined_results = combine_aime_scores(results)

    # Models to compare - show ft models from 29 to 928 examples
    models = ['base', 'lora', 'ft-29', 'ft-58', 'ft-116', 'ft-232', 'ft-464', 'ft-928', 'deepseek']
    model_labels = ['Base', 'LoRA', 'FT-29', 'FT-58', 'FT-116', 'FT-232', 'FT-464', 'FT-928', 'DeepSeek']

    # Model sizes to show
    model_sizes = ['1.5B', '7B', '14B', '32B']

    # Colors for each model variant
    colors = {
        'base': '#808080',      # Gray
        'lora': '#FFA500',      # Orange
        'ft-29': '#E8D5C4',     # Light purple
        'ft-58': '#D7B5C8',     # Light-medium purple
        'ft-116': '#C495CC',    # Medium purple
        'ft-232': '#B175D0',    # Medium-dark purple
        'ft-464': '#9E55D4',    # Dark purple
        'ft-928': '#A23B72',    # Original purple
        'deepseek': '#2A9D8F',  # Teal
    }

    # Create figure with 1x2 subplots (wider to accommodate more bars)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Width of bars and positions (smaller width for more models)
    bar_width = 0.09
    x_positions = np.arange(len(model_sizes))

    # Collect AIME scores for all model sizes
    aime_scores_by_model = {model: [] for model in models}
    aime_errors_by_model = {model: [] for model in models}

    for size in model_sizes:
        for model in models:
            if model == 'lora':
                # Get LoRA AIME scores
                if size in combined_results and 'lora' in combined_results[size]:
                    score = combined_results[size]['lora'].get('aime_combined', {}).get('score', 0) * 100
                    stderr = combined_results[size]['lora'].get('aime_combined', {}).get('stderr', 0) * 100
                else:
                    score = 0
                    stderr = 0
            else:
                # Get regular model scores
                score = combined_results.get(size, {}).get(model, {}).get('aime_combined', {}).get('score', 0) * 100
                stderr = combined_results.get(size, {}).get(model, {}).get('aime_combined', {}).get('stderr', 0) * 100

            aime_scores_by_model[model].append(score)
            aime_errors_by_model[model].append(stderr)

    # Collect GPQA scores for all model sizes
    gpqa_scores_by_model = {model: [] for model in models}
    gpqa_errors_by_model = {model: [] for model in models}

    for size in model_sizes:
        for model in models:
            if model == 'lora':
                # Get LoRA GPQA scores from multiple runs
                scores = []
                dirs = [
                    "results/temperature",
                    "results/temperature_gpqa_2",
                    "results/temperature_gpqa_3",
                    "results/temperature_gpqa_4"
                ]

                for dir_name in dirs:
                    dir_path = Path(dir_name) / f"{size}-lora"
                    if dir_path.exists():
                        for subdir in dir_path.iterdir():
                            if subdir.is_dir():
                                json_files = list(subdir.glob("results_*.json"))
                                if json_files:
                                    json_file = sorted(json_files)[-1]
                                    try:
                                        with open(json_file, 'r') as f:
                                            data = json.load(f)
                                        if "results" in data and "gpqa_diamond_cot_zeroshot" in data["results"]:
                                            score_val = data["results"]["gpqa_diamond_cot_zeroshot"].get("exact_match,none", None)
                                            if score_val is not None:
                                                scores.append(score_val)
                                                break
                                    except Exception:
                                        pass

                if scores:
                    mean_score = np.mean(scores) * 100
                    if len(scores) > 1:
                        std_score = np.std(scores, ddof=1) * 100
                        stderr = std_score / np.sqrt(len(scores))
                    else:
                        stderr = 0
                    gpqa_scores_by_model[model].append(mean_score)
                    gpqa_errors_by_model[model].append(stderr)
                else:
                    gpqa_scores_by_model[model].append(0)
                    gpqa_errors_by_model[model].append(0)
            else:
                # Use existing parse_gpqa_multiple_runs function for other models
                mean, stderr = parse_gpqa_multiple_runs(size, model)
                gpqa_scores_by_model[model].append(mean * 100)
                gpqa_errors_by_model[model].append(stderr * 100)

    # Plot 1: AIME (left)
    ax = axes[0]

    # Create grouped bars for each model
    for i, model in enumerate(models):
        positions = x_positions + i * bar_width - bar_width * (len(models) - 1) / 2  # Center the groups
        bars = ax.bar(positions, aime_scores_by_model[model], bar_width,
                      label=model_labels[i],
                      color=colors[model],
                      yerr=aime_errors_by_model[model],
                      edgecolor='black', linewidth=0.5, capsize=3)

        # Add value labels on bars
        for bar, score in zip(bars, aime_scores_by_model[model]):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{score:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Absolute performance (%)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_sizes)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9)

    # Plot 2: GPQA (right)
    ax = axes[1]

    # Create grouped bars for each model
    for i, model in enumerate(models):
        positions = x_positions + i * bar_width - bar_width * (len(models) - 1) / 2  # Center the groups
        bars = ax.bar(positions, gpqa_scores_by_model[model], bar_width,
                      label=model_labels[i],
                      color=colors[model],
                      yerr=gpqa_errors_by_model[model],
                      edgecolor='black', linewidth=0.5, capsize=3)

        # Add value labels on bars
        for bar, score in zip(bars, gpqa_scores_by_model[model]):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{score:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Absolute performance (%)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_sizes)
    ax.set_ylim(0, 70)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9)

    # Overall title
    # plt.suptitle('32B Model Performance Comparison: LoRA vs Fine-tuning',
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    plot_path = PLOTS_DIR / 'lora_comparison_sidebyside.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def main():
    """Main function to run LoRA performance plotting"""
    from data_utils import parse_results_directory, save_parsed_data

    # Parse all results
    results = parse_results_directory()

    # Save parsed data
    save_parsed_data(results)

    # Create LoRA comparison plots
    print("\nCreating LoRA comparison plots...")
    create_lora_comparison_bar_plot(results)
    create_lora_comparison_sidebyside_bar_plot(results)

    print("\nAll LoRA plots created successfully!")


if __name__ == "__main__":
    main()