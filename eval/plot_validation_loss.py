#!/usr/bin/env python3
"""
Plot validation loss vs performance scatter plots
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from data_utils import (
    combine_aime_scores,
    PLOTS_DIR
)


def create_loss_vs_performance_scatter_plot(results: Dict):
    """Create scatter plot showing validation loss decrease vs performance improvement"""

    # Combine AIME scores first
    combined_results = combine_aime_scores(results)

    # Define model sizes and dataset sizes
    model_sizes = ['1.5B', '7B', '14B', '32B']
    dataset_sizes = ['928', '464', '232', '116', '58', '29']

    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Define colors for model sizes
    model_colors = {
        '1.5B': '#FF6B6B',
        '7B': '#4ECDC4',
        '14B': '#2E86AB',
        '32B': '#A23B72'
    }

    # Define markers for dataset sizes (matching recovery plot)
    dataset_markers = {
        '928': '>',  # right triangle
        '464': '^',  # triangle up
        '232': '<',  # left triangle
        '116': 'v',  # down triangle
        '58': 's',   # square
        '29': 'p'    # pentagon
    }

    # Collect data for both plots
    aime_data = {'x': [], 'y': [], 'model': [], 'dataset': []}
    gpqa_data = {'x': [], 'y': [], 'model': [], 'dataset': []}

    for model_size in model_sizes:
        for dataset_size in dataset_sizes:
            # Get validation loss improvement
            pattern = f"{model_size}-{dataset_size}-val72-*"
            dirs = glob.glob(f"/workspace/s1_peft/ckpts_sample/{pattern}")
            dirs = [d for d in dirs if not d.endswith('_final')]

            if dirs:
                metrics_file = Path(dirs[0]) / "final_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)

                    # Calculate validation loss improvement (decrease)
                    loss_improvement = metrics['initial_eval_loss'] - metrics['best_eval_loss']

                    # Get AIME performance improvement
                    variant = f'ft-{dataset_size}'
                    base_aime = combined_results.get(model_size, {}).get('base', {}).get('aime_combined', {}).get('score', 0)
                    ft_aime = combined_results.get(model_size, {}).get(variant, {}).get('aime_combined', {}).get('score', 0)
                    aime_improvement = (ft_aime - base_aime) * 100  # Convert to percentage points

                    # Get GPQA performance improvement
                    base_gpqa = results.get(model_size, {}).get('base', {}).get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
                    ft_gpqa = results.get(model_size, {}).get(variant, {}).get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
                    gpqa_improvement = (ft_gpqa - base_gpqa) * 100  # Convert to percentage points

                    # Store data
                    aime_data['x'].append(loss_improvement)
                    aime_data['y'].append(aime_improvement)
                    aime_data['model'].append(model_size)
                    aime_data['dataset'].append(dataset_size)

                    gpqa_data['x'].append(loss_improvement)
                    gpqa_data['y'].append(gpqa_improvement)
                    gpqa_data['model'].append(model_size)
                    gpqa_data['dataset'].append(dataset_size)

    # Plot 1: AIME (left)
    ax = axes[0]

    # Create custom legend handles
    model_handles = []
    dataset_handles = []

    # First, plot the individual data points with reduced prominence
    for i in range(len(aime_data['x'])):
        ax.scatter(aime_data['x'][i], aime_data['y'][i],
                  c=model_colors[aime_data['model'][i]],
                  marker=dataset_markers[aime_data['dataset'][i]],
                  s=80, alpha=0.3, edgecolors='none', zorder=1)

    # Then calculate and plot centroids for each model size
    for model_size in model_sizes:
        # Collect points for this model size
        x_vals = []
        y_vals = []
        for i in range(len(aime_data['x'])):
            if aime_data['model'][i] == model_size:
                x_vals.append(aime_data['x'][i])
                y_vals.append(aime_data['y'][i])

        if x_vals:  # If we have data for this model size
            # Calculate centroid
            centroid_x = np.mean(x_vals)
            centroid_y = np.mean(y_vals)

            # Plot centroid as a larger, prominent marker
            ax.scatter(centroid_x, centroid_y,
                      c=model_colors[model_size],
                      marker='x',  # X marker for centroids
                      s=300, alpha=1.0, linewidths=3,
                      zorder=2, label=f'{model_size} centroid')

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)  # Darker y=0 line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Validation Loss Decrease (nats)', fontsize=11)
    ax.set_ylabel('Performance Improvement (pp)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend for model sizes (colors)
    for model_size, color in model_colors.items():
        model_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=8,
                                       label=model_size))

    # Create legend for dataset sizes (markers)
    for dataset_size, marker in dataset_markers.items():
        dataset_handles.append(plt.Line2D([0], [0], marker=marker, color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label=f'{dataset_size} examples'))

    # Add X marker to indicate centroids
    dataset_handles.append(plt.Line2D([0], [0], marker='x', color='gray',
                                     markersize=10, markeredgewidth=2,
                                     label='Centroid', linestyle='none'))

    # Add both legends
    legend1 = ax.legend(handles=model_handles, loc='upper left', title='Model Size', fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, loc='lower left', title='Dataset Size', fontsize=8)

    # Plot 2: GPQA (right)
    ax = axes[1]

    # First, plot the individual data points with reduced prominence
    for i in range(len(gpqa_data['x'])):
        ax.scatter(gpqa_data['x'][i], gpqa_data['y'][i],
                  c=model_colors[gpqa_data['model'][i]],
                  marker=dataset_markers[gpqa_data['dataset'][i]],
                  s=80, alpha=0.3, edgecolors='none', zorder=1)

    # Then calculate and plot centroids for each model size
    for model_size in model_sizes:
        # Collect points for this model size
        x_vals = []
        y_vals = []
        for i in range(len(gpqa_data['x'])):
            if gpqa_data['model'][i] == model_size:
                x_vals.append(gpqa_data['x'][i])
                y_vals.append(gpqa_data['y'][i])

        if x_vals:  # If we have data for this model size
            # Calculate centroid
            centroid_x = np.mean(x_vals)
            centroid_y = np.mean(y_vals)

            # Plot centroid as a larger, prominent marker
            ax.scatter(centroid_x, centroid_y,
                      c=model_colors[model_size],
                      marker='x',  # X marker for centroids
                      s=300, alpha=1.0, linewidths=3,
                      zorder=2)

    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)  # Darker y=0 line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Validation Loss Decrease (nats)', fontsize=11)
    ax.set_ylabel('Performance Improvement (pp)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add legends to second plot as well
    legend1 = ax.legend(handles=model_handles, loc='upper left', title='Model Size', fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=dataset_handles, loc='lower left', title='Dataset Size', fontsize=8)

    # Overall title
    # plt.suptitle('Validation Loss vs Performance Improvement: Revealing the Learning-Reasoning Disconnect',
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save the figure
    plot_path = PLOTS_DIR / 'loss_vs_performance_scatter.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def main():
    """Main function to run validation loss plotting"""
    from data_utils import parse_results_directory, save_parsed_data

    # Parse all results
    results = parse_results_directory()

    # Save parsed data
    save_parsed_data(results)

    # Create validation loss scatter plot
    print("\nCreating validation loss vs performance scatter plot...")
    create_loss_vs_performance_scatter_plot(results)

    print("\nValidation loss plot created successfully!")


if __name__ == "__main__":
    main()