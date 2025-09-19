#!/usr/bin/env python3
"""
Performance recovery and absolute performance plotting functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from pathlib import Path
from typing import Dict

from data_utils import (
    get_variant_label,
    parse_results_directory,
    combine_aime_scores,
    calculate_performance_recovered,
    PLOTS_DIR
)


def create_absolute_performance_graphs(results: Dict):
    """Create line graphs showing actual performance with shaded error regions"""
    
    # Combine AIME scores first
    results = combine_aime_scores(results)
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Create figure with single subplot for combined AIME
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for each variant
    colors = {
        'base': '#808080',
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#FF5722',     # Deep Orange
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',
        'ft-232': '#4ECDC4',
        'ft-464': '#2E86AB',
        'ft-928': '#A23B72',
        'deepseek': '#2A9D8F'
    }
    
    markers = {
        'base': 'o',
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': '*',       # star
        'ft-58': 'x',       # x marker
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
        'deepseek': 'D'
    }
    
    # Plot combined AIME scores
    for variant in ['deepseek', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7', 'base']:
        scores = []
        stderrs = []
        valid_sizes = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if size in results and variant in results[size] and 'aime_combined' in results[size][variant]:
                scores.append(results[size][variant]['aime_combined']['score'])
                stderrs.append(results[size][variant]['aime_combined']['stderr'])
                valid_sizes.append(size)
                valid_numeric.append(size_numeric[i])
        
        if scores:
            scores = np.array(scores)
            stderrs = np.array(stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            ax.plot(valid_numeric, scores, 
                    marker=markers[variant], 
                    label=get_variant_label(variant),
                    linewidth=2, markersize=8, color=colors[variant])
            
            # Add shaded error region (using standard error)
            ax.fill_between(valid_numeric, 
                            scores - stderrs, 
                            scores + stderrs,
                            alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('AIME Score (Combined 2024 & 2025)', fontsize=12)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(-0.05, 0.7)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # plt.suptitle('Absolute Performance Across Model Sizes (with Standard Errors)', 
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'new_absolute_performance.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()

def create_performance_recovery_graphs(results: Dict):
    """Create line graphs showing performance recovery with shaded error regions"""
    
    # Combine AIME scores first
    results = combine_aime_scores(results)
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Create figure with single subplot for combined AIME
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for FT variants and LoRA
    colors = {
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#FF5722',     # Deep Orange
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',
        'ft-232': '#4ECDC4',
        'ft-464': '#2E86AB',
        'ft-928': '#A23B72',
        'lora': '#FFA500',  # Orange for LoRA
    }
    
    markers = {
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': '*',       # star
        'ft-58': 'x',       # x marker
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
        'lora': 'D',  # Diamond for LoRA
    }
    
    # Plot Combined AIME Performance Recovery
    for variant in ['ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'lora']:
        recovery_scores = []
        recovery_stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if (size in results and 
                variant in results[size] and 
                'base' in results[size] and 
                'deepseek' in results[size] and
                'aime_combined' in results[size][variant]):
                
                ft_score = results[size][variant]['aime_combined']['score']
                ft_stderr = results[size][variant]['aime_combined']['stderr']
                base_score = results[size]['base']['aime_combined']['score']
                base_stderr = results[size]['base']['aime_combined']['stderr']
                deepseek_score = results[size]['deepseek']['aime_combined']['score']
                deepseek_stderr = results[size]['deepseek']['aime_combined']['stderr']
                
                # Calculate recovery percentage
                recovery, clamped = calculate_performance_recovered(ft_score, base_score, deepseek_score)
                recovery_scores.append(recovery)
                
                # Propagate error for recovery calculation (simplified)
                # Using approximation for error propagation in division
                if not clamped:
                    denominator = deepseek_score - base_score
                    # Simplified error propagation
                    relative_stderr = ft_stderr / abs(denominator) * 100
                    recovery_stderrs.append(relative_stderr)
                else:
                    recovery_stderrs.append(0)
                
                valid_numeric.append(size_numeric[i])
        
        if recovery_scores:
            recovery_scores = np.array(recovery_scores)
            recovery_stderrs = np.array(recovery_stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            label = get_variant_label(variant)
            ax.plot(valid_numeric, recovery_scores,
                    marker=markers[variant],
                    label=label,
                    linewidth=2, markersize=8, color=colors[variant])
            
            # Add shaded error region (using standard error)
            ax.fill_between(valid_numeric,
                            recovery_scores - recovery_stderrs,
                            recovery_scores + recovery_stderrs,
                            alpha=0.2, color=colors[variant])
    
    # Add reference lines
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Deepseek-R1-Distill-Qwen (skyline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Performance Recovered (%)', fontsize=12)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(-20, 120)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # plt.suptitle('Fine-tuning Performance Recovery vs Model Size (with Standard Errors)',
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'new_performance_recovery.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()

def create_gpqa_absolute_performance_graphs(results: Dict):
    """Create line graphs showing GPQA absolute performance with error bars"""
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Check if we have any GPQA data
    has_gpqa = False
    for size in results.values():
        for variant in size.values():
            if 'gpqa_diamond_cot_zeroshot' in variant:
                has_gpqa = True
                break
        if has_gpqa:
            break
    
    if not has_gpqa:
        print("No GPQA data found, skipping GPQA plots")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for each variant
    colors = {
        'base': '#808080',
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#FF5722',     # Deep Orange
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',
        'ft-232': '#4ECDC4',
        'ft-464': '#2E86AB',
        'ft-928': '#A23B72',
        'deepseek': '#2A9D8F'
    }
    
    markers = {
        'base': 'o',
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': '*',       # star
        'ft-58': 'x',       # x marker
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
        'deepseek': 'D'
    }
    
    # Plot GPQA scores
    for variant in ['deepseek', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7', 'base']:
        scores = []
        stderrs = []
        valid_sizes = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if size in results and variant in results[size] and 'gpqa_diamond_cot_zeroshot' in results[size][variant]:
                scores.append(results[size][variant]['gpqa_diamond_cot_zeroshot']['score'])
                stderrs.append(results[size][variant]['gpqa_diamond_cot_zeroshot']['stderr'])
                valid_sizes.append(size)
                valid_numeric.append(size_numeric[i])
        
        if scores:
            scores = np.array(scores)
            stderrs = np.array(stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            ax.plot(valid_numeric, scores, 
                    marker=markers[variant], 
                    label=get_variant_label(variant),
                    linewidth=2, markersize=8, color=colors[variant])
            
            # Add shaded error region
            ax.fill_between(valid_numeric, 
                            scores - stderrs, 
                            scores + stderrs,
                            alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('GPQA Accuracy', fontsize=12)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # plt.suptitle('GPQA Absolute Performance Across Model Sizes', 
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'gpqa_absolute_performance.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def create_gpqa_performance_recovery_graphs(results: Dict):
    """Create line graphs showing GPQA performance recovery"""
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Check if we have GPQA data
    has_gpqa = False
    for size in results.values():
        for variant in size.values():
            if 'gpqa_diamond_cot_zeroshot' in variant:
                has_gpqa = True
                break
        if has_gpqa:
            break
    
    if not has_gpqa:
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for FT variants
    colors = {
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#FF5722',     # Deep Orange
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',
        'ft-232': '#4ECDC4',
        'ft-464': '#2E86AB',
        'ft-928': '#A23B72',
    }
    
    markers = {
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': '*',       # star
        'ft-58': 'x',       # x marker
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
    }
    
    # Plot GPQA Performance Recovery
    for variant in ['ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7']:
        recovery_scores = []
        recovery_stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if (size in results and 
                variant in results[size] and 
                'base' in results[size] and 
                'deepseek' in results[size] and
                'gpqa_diamond_cot_zeroshot' in results[size][variant]):
                
                ft_score = results[size][variant]['gpqa_diamond_cot_zeroshot']['score']
                ft_stderr = results[size][variant]['gpqa_diamond_cot_zeroshot']['stderr']
                base_score = results[size]['base'].get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
                base_stderr = results[size]['base'].get('gpqa_diamond_cot_zeroshot', {}).get('stderr', 0)
                deepseek_score = results[size]['deepseek'].get('gpqa_diamond_cot_zeroshot', {}).get('score', 1)
                deepseek_stderr = results[size]['deepseek'].get('gpqa_diamond_cot_zeroshot', {}).get('stderr', 0)
                
                # Calculate recovery percentage
                recovery, clamped = calculate_performance_recovered(ft_score, base_score, deepseek_score)
                recovery_scores.append(recovery)
                
                # Propagate error for recovery calculation
                if not clamped:
                    denominator = deepseek_score - base_score
                    relative_stderr = ft_stderr / abs(denominator) * 100
                    recovery_stderrs.append(relative_stderr)
                else:
                    recovery_stderrs.append(0)
                
                valid_numeric.append(size_numeric[i])
        
        if recovery_scores:
            recovery_scores = np.array(recovery_scores)
            recovery_stderrs = np.array(recovery_stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            ax.plot(valid_numeric, recovery_scores,
                    marker=markers[variant],
                    label=get_variant_label(variant),
                    linewidth=2, markersize=8, color=colors[variant])
            
            # Add shaded error region
            ax.fill_between(valid_numeric,
                            recovery_scores - recovery_stderrs,
                            recovery_scores + recovery_stderrs,
                            alpha=0.2, color=colors[variant])
    
    # Add reference lines
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Deepseek-R1-Distill-Qwen (skyline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    ax.set_xlabel('Model Size', fontsize=12)
    ax.set_ylabel('Performance Recovered (%)', fontsize=12)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(-20, 120)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # plt.suptitle('GPQA Fine-tuning Performance Recovery vs Model Size',
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'gpqa_performance_recovery.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def create_combined_1x2_plot(results: Dict):
    """Create a 1x2 plot with Combined AIME and GPQA"""
    
    # Combine AIME scores first for the combined plot
    combined_results = combine_aime_scores(results)
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Define colors with gradient-like progression
    colors = {
        'base': '#808080',
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#C0C0C0',     # Light gray
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF4500',    # Dark orange/OrangeRed
        'ft-232': '#4ECDC4',    # Cyan
        'ft-464': '#2E86AB',    # Blue
        'ft-928': '#A23B72',    # Purple
        'deepseek': '#2A9D8F',  # Teal
        'lora': '#000000'       # Black to stand out
    }
    
    markers = {
        'base': 'o',
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': 'p',       # pentagon
        'ft-58': 's',       # square
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
        'deepseek': 'D',
        'lora': 'P'  # Plus marker for LoRA
    }
    
    # Plot 1: Combined AIME (left) with bootstrap stderr
    ax = axes[0]
    for variant in ['deepseek', 'base', 'ft-928', 'lora']:
        scores = []
        stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if size in combined_results and variant in combined_results[size] and 'aime_combined' in combined_results[size][variant]:
                scores.append(combined_results[size][variant]['aime_combined']['score'])
                stderrs.append(combined_results[size][variant]['aime_combined']['stderr'])
                valid_numeric.append(size_numeric[i])
        
        if scores:
            scores = np.array(scores)
            stderrs = np.array(stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            label = get_variant_label(variant)
            # Use different line styles: dotted for FT, dashed for base/skyline, solid for LoRA
            if variant.startswith('ft-'):
                linestyle = ':'  # Dotted for full finetunes
            elif variant in ['deepseek', 'base']:
                linestyle = '--'  # Dashed for base/skyline
            else:  # variant == 'lora'
                linestyle = '-'  # Solid for LoRA
            ax.plot(valid_numeric, scores, 
                    marker=markers[variant], 
                    label=label,
                    linewidth=2, markersize=8, color=colors[variant], linestyle=linestyle)
            
            # Add shaded error region
            ax.fill_between(valid_numeric, 
                            scores - stderrs, 
                            scores + stderrs,
                            alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('AIME Score', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(-0.05, 0.7)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
    
    # Plot 2: GPQA (right) with existing stderr
    ax = axes[1]
    for variant in ['deepseek', 'base', 'ft-928', 'lora']:
        scores = []
        stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            if size in results and variant in results[size] and 'gpqa_diamond_cot_zeroshot' in results[size][variant]:
                scores.append(results[size][variant]['gpqa_diamond_cot_zeroshot']['score'])
                stderrs.append(results[size][variant]['gpqa_diamond_cot_zeroshot']['stderr'])
                valid_numeric.append(size_numeric[i])
        
        if scores:
            scores = np.array(scores)
            stderrs = np.array(stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            label = get_variant_label(variant)
            # Use different line styles: dotted for FT, dashed for base/skyline, solid for LoRA
            if variant.startswith('ft-'):
                linestyle = ':'  # Dotted for full finetunes
            elif variant in ['deepseek', 'base']:
                linestyle = '--'  # Dashed for base/skyline
            else:  # variant == 'lora'
                linestyle = '-'  # Solid for LoRA
            ax.plot(valid_numeric, scores, 
                    marker=markers[variant], 
                    label=label,
                    linewidth=2, markersize=8, color=colors[variant], linestyle=linestyle)
            
            # Add shaded error region
            ax.fill_between(valid_numeric, 
                            scores - stderrs, 
                            scores + stderrs,
                            alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('GPQA Accuracy', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(0.08, 0.62)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
    
    # Overall title
    # plt.suptitle('Model Performance Across Benchmarks (Temperature Sampling)', 
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'temperature_1x2_performance.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def create_recovery_1x2_plot(results: Dict):
    """Create a 1x2 plot showing performance recovery for AIME and GPQA"""
    
    # Combine AIME scores first for the combined plot
    combined_results = combine_aime_scores(results)
    
    # Define sizes and their numeric values for plotting
    sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define colors for FT variants and LoRA
    colors = {
        'ft-7': '#FFC107',      # Amber
        'ft-14': '#FF9800',     # Orange
        'ft-29': '#C0C0C0',     # Light gray (matching performance plot)
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',
        'ft-232': '#4ECDC4',
        'ft-464': '#2E86AB',
        'ft-928': '#A23B72',
        'lora': '#FFA500',  # Orange for LoRA
    }
    
    markers = {
        'ft-7': 'p',        # pentagon
        'ft-14': 'h',       # hexagon
        'ft-29': 'p',       # pentagon (matching performance plot)
        'ft-58': 's',       # square (matching performance plot)
        'ft-116': 'v',
        'ft-232': '<',
        'ft-464': '^',
        'ft-928': '>',  # right triangle
        'lora': 'P',  # Plus marker for LoRA
    }
    
    # Plot 1: Combined AIME Recovery (left)
    ax = axes[0]
    
    # Add reference lines first so they appear first in legend
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Deepseek-R1-Distill-Qwen (skyline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    for variant in ['ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29']:
        recovery_scores = []
        recovery_stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(sizes):
            # Check if we have all necessary data for recovery calculation
            if (size in combined_results and 
                variant in combined_results[size] and 
                'base' in combined_results[size] and 
                'deepseek' in combined_results[size] and
                'aime_combined' in combined_results[size][variant]):
                
                ft_score = combined_results[size][variant]['aime_combined']['score']
                ft_stderr = combined_results[size][variant]['aime_combined']['stderr']
                base_score = combined_results[size]['base']['aime_combined']['score']
                deepseek_score = combined_results[size]['deepseek']['aime_combined']['score']
                
                # Calculate recovery percentage
                recovery, is_valid = calculate_performance_recovered(ft_score, base_score, deepseek_score)
                recovery_scores.append(recovery)

                # Only calculate error if valid
                if is_valid:
                    denominator = deepseek_score - base_score
                    relative_stderr = ft_stderr / abs(denominator) * 100
                    recovery_stderrs.append(relative_stderr)
                else:
                    recovery_stderrs.append(0)
                
                valid_numeric.append(size_numeric[i])
        
        if recovery_scores:
            recovery_scores = np.array(recovery_scores)
            recovery_stderrs = np.array(recovery_stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Plot line with markers
            label = get_variant_label(variant)
            ax.plot(valid_numeric, recovery_scores,
                    marker=markers[variant],
                    label=label,
                    linewidth=2, markersize=8, color=colors[variant])

            # Add shaded error region
            ax.fill_between(valid_numeric,
                            recovery_scores - recovery_stderrs,
                            recovery_scores + recovery_stderrs,
                            alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Performance Recovered (%)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(sizes)
    ax.set_xlim(1, 50)
    ax.set_ylim(-30, 120)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=9)
    
    # Plot 2: GPQA Recovery (right) - Only 7B, 14B, 32B
    ax = axes[1]
    # Use only 7B, 14B, 32B for GPQA to avoid clamped 1.5B data
    gpqa_sizes = ['7B', '14B', '32B']
    gpqa_size_numeric = [7, 14, 32]
    
    # Add reference lines first so they appear first in legend
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Deepseek-R1-Distill-Qwen (skyline)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    for variant in ['ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29']:
        recovery_scores = []
        recovery_stderrs = []
        valid_numeric = []
        was_clamped = []  # Track which points were clamped
        
        for i, size in enumerate(gpqa_sizes):
            # Check if we have all necessary data for recovery calculation
            if (size in results and 
                variant in results[size] and 
                'base' in results[size] and 
                'deepseek' in results[size] and
                'gpqa_diamond_cot_zeroshot' in results[size][variant] and
                'gpqa_diamond_cot_zeroshot' in results[size]['base'] and
                'gpqa_diamond_cot_zeroshot' in results[size]['deepseek']):
                
                ft_score = results[size][variant]['gpqa_diamond_cot_zeroshot']['score']
                ft_stderr = results[size][variant]['gpqa_diamond_cot_zeroshot']['stderr']
                base_score = results[size]['base']['gpqa_diamond_cot_zeroshot']['score']
                deepseek_score = results[size]['deepseek']['gpqa_diamond_cot_zeroshot']['score']
                
                # Calculate recovery percentage with validity flag
                recovery, is_valid = calculate_performance_recovered(ft_score, base_score, deepseek_score)
                recovery_scores.append(recovery)
                was_clamped.append(not is_valid)  # Clamped when NOT valid
                
                # Only calculate error if valid (not clamped)
                if is_valid:
                    denominator = deepseek_score - base_score
                    relative_stderr = ft_stderr / abs(denominator) * 100
                    recovery_stderrs.append(relative_stderr)
                else:
                    recovery_stderrs.append(0)  # No error bar for clamped values
                
                valid_numeric.append(gpqa_size_numeric[i])
        
        if recovery_scores:
            recovery_scores = np.array(recovery_scores)
            recovery_stderrs = np.array(recovery_stderrs)
            valid_numeric = np.array(valid_numeric)
            was_clamped = np.array(was_clamped)
            
            # Plot line segments with different styles for clamped sections
            for j in range(len(valid_numeric)):
                if j == 0:
                    # First point - just plot marker
                    marker_style = 'o' if was_clamped[j] else markers[variant]  # Hollow circle if clamped
                    ax.plot(valid_numeric[j], recovery_scores[j],
                            marker=marker_style,
                            markersize=8, color=colors[variant],
                            markerfacecolor='white' if was_clamped[j] else colors[variant],
                            markeredgecolor=colors[variant], markeredgewidth=2)
                else:
                    # Draw line segment from previous point (always solid)
                    ax.plot(valid_numeric[j-1:j+1], recovery_scores[j-1:j+1],
                            linestyle='-', linewidth=2, color=colors[variant])
                    
                    # Draw marker
                    marker_style = 'o' if was_clamped[j] else markers[variant]
                    ax.plot(valid_numeric[j], recovery_scores[j],
                            marker=marker_style,
                            markersize=8, color=colors[variant],
                            markerfacecolor='white' if was_clamped[j] else colors[variant],
                            markeredgecolor=colors[variant], markeredgewidth=2)
            
            # Add label once
            label = get_variant_label(variant)
            ax.plot([], [], marker=markers[variant], color=colors[variant],
                    label=label,
                    linewidth=2, markersize=8)
            
            # Add shaded error regions for continuous non-clamped segments
            # Find continuous segments of non-clamped values
            segment_start = None
            for j in range(len(valid_numeric)):
                if not was_clamped[j]:
                    if segment_start is None:
                        segment_start = j
                else:
                    # End of non-clamped segment
                    if segment_start is not None and j > segment_start:
                        # Draw error region for this segment
                        segment_x = valid_numeric[segment_start:j]
                        segment_y = recovery_scores[segment_start:j]
                        segment_err = recovery_stderrs[segment_start:j]
                        ax.fill_between(segment_x,
                                        segment_y - segment_err,
                                        segment_y + segment_err,
                                        alpha=0.2, color=colors[variant])
                    segment_start = None
            
            # Handle case where last segment goes to the end
            if segment_start is not None:
                segment_x = valid_numeric[segment_start:]
                segment_y = recovery_scores[segment_start:]
                segment_err = recovery_stderrs[segment_start:]
                ax.fill_between(segment_x,
                                segment_y - segment_err,
                                segment_y + segment_err,
                                alpha=0.2, color=colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Performance Recovered (%)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(gpqa_size_numeric)
    ax.set_xticklabels(gpqa_sizes)
    ax.set_xticks([], minor=True)  # Remove minor ticks
    ax.set_xlim(5, 50)
    ax.set_ylim(-80, 140)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=9)
    
    # Overall title with note about clamped values
    # plt.suptitle('Performance Recovery Across Benchmarks (% of DeepSeek Performance)', 
    #              fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure in both PDF and PNG formats
    plot_path_pdf = PLOTS_DIR / 'temperature_recovery_1x2.pdf'
    plot_path_png = PLOTS_DIR / 'temperature_recovery_1x2.png'
    plt.savefig(plot_path_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path_pdf}")
    print(f"Graph saved as: {plot_path_png}")
    plt.show()


# Removed create_absolute_lift_1x2_plot - replaced by create_combined_improvement_1x2_plot
def create_combined_improvement_1x2_plot(results: Dict):
    """Create a 1x2 plot showing absolute performance lift lines for AIME and GPQA"""
    
    # Combine AIME scores first
    combined_results = combine_aime_scores(results)
    
    # Define model sizes and dataset sizes
    model_sizes = ['1.5B', '7B', '14B', '32B']
    size_numeric = [1.5, 7, 14, 32]
    dataset_sizes = ['29', '58', '116', '232', '464', '928']
    x_positions = np.arange(len(model_sizes))
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Define colors for line plots (matching recovery plot style)
    line_colors = {
        'deepseek': '#2A9D8F',  # Teal
        'ft-29': '#C0C0C0',     # Light gray
        'ft-58': '#FFB6C1',     # Light pink
        'ft-116': '#FF6B6B',    # Light red
        'ft-232': '#4ECDC4',    # Cyan
        'ft-464': '#2E86AB',    # Blue
        'ft-928': '#A23B72',    # Purple
    }
    
    # Define markers for line plots
    line_markers = {
        'deepseek': 'D',        # Diamond
        'ft-29': 'p',           # Pentagon
        'ft-58': 's',           # Square
        'ft-116': 'v',          # Down triangle
        'ft-232': '<',          # Left triangle
        'ft-464': '^',          # Up triangle
        'ft-928': '>',          # Right triangle
    }
    
    # Variants to plot
    variants = ['deepseek', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29']
    
    # ============================================
    # Plot 1: Combined AIME Performance (left) - Line Plot
    # ============================================
    ax = axes[0]
    
    # Add baseline reference line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    # Plot lines for each variant
    for variant in variants:
        lift_scores = []
        lift_stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(model_sizes):
            # Get base score
            base_score = combined_results.get(size, {}).get('base', {}).get('aime_combined', {}).get('score', 0)
            
            # Get variant score (including deepseek)
            variant_data = combined_results.get(size, {}).get(variant, {}).get('aime_combined', {})
            
            if variant_data:
                variant_score = variant_data.get('score', 0)
                variant_stderr = variant_data.get('stderr', 0)
                
                # Calculate absolute lift in percentage points
                lift = (variant_score - base_score) * 100
                lift_scores.append(lift)
                lift_stderrs.append(variant_stderr * 100)
                valid_numeric.append(size_numeric[i])
        
        if lift_scores:
            lift_scores = np.array(lift_scores)
            lift_stderrs = np.array(lift_stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Get label
            if variant == 'deepseek':
                label = 'DeepSeek-R1-Distill-Qwen'
            else:
                label = get_variant_label(variant)
            
            # Plot line with markers (dotted for deepseek)
            linestyle = '--' if variant == 'deepseek' else '-'
            ax.plot(valid_numeric, lift_scores,
                    marker=line_markers[variant],
                    label=label,
                    linewidth=2, markersize=8, color=line_colors[variant],
                    linestyle=linestyle)
            
            # Add shaded error region
            ax.fill_between(valid_numeric,
                            lift_scores - lift_stderrs,
                            lift_scores + lift_stderrs,
                            alpha=0.2, color=line_colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Absolute Performance Lift (pp)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(size_numeric)
    ax.set_xticklabels(model_sizes)
    ax.set_xlim(1, 50)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-5, 55)
    
    # ============================================
    # Plot 2: GPQA Performance (right) - Line Plot
    # ============================================
    ax = axes[1]
    
    # Use all model sizes including 1.5B
    gpqa_sizes = model_sizes
    gpqa_size_numeric = size_numeric
    
    # Add baseline reference line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Qwen2.5-Instruct (baseline)')
    
    # Plot lines for each variant
    for variant in variants:
        lift_scores = []
        lift_stderrs = []
        valid_numeric = []
        
        for i, size in enumerate(gpqa_sizes):
            # Get base score
            base_score = results.get(size, {}).get('base', {}).get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
            
            # Get variant score (including deepseek)
            variant_data = results.get(size, {}).get(variant, {}).get('gpqa_diamond_cot_zeroshot', {})
            
            if variant_data:
                variant_score = variant_data.get('score', 0)
                variant_stderr = variant_data.get('stderr', 0)
                
                # Calculate absolute lift in percentage points
                lift = (variant_score - base_score) * 100
                lift_scores.append(lift)
                lift_stderrs.append(variant_stderr * 100)
                valid_numeric.append(gpqa_size_numeric[i])
        
        if lift_scores:
            lift_scores = np.array(lift_scores)
            lift_stderrs = np.array(lift_stderrs)
            valid_numeric = np.array(valid_numeric)
            
            # Get label
            if variant == 'deepseek':
                label = 'DeepSeek-R1-Distill-Qwen'
            else:
                label = get_variant_label(variant)
            
            # Plot line with markers (dotted for deepseek)
            linestyle = '--' if variant == 'deepseek' else '-'
            ax.plot(valid_numeric, lift_scores,
                    marker=line_markers[variant],
                    label=label,
                    linewidth=2, markersize=8, color=line_colors[variant],
                    linestyle=linestyle)
            
            # Add shaded error region
            ax.fill_between(valid_numeric,
                            lift_scores - lift_stderrs,
                            lift_scores + lift_stderrs,
                            alpha=0.2, color=line_colors[variant])
    
    ax.set_xlabel('Model Size', fontsize=11)
    ax.set_ylabel('Absolute Performance Lift (pp)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log', base=10)
    ax.set_xticks(gpqa_size_numeric)
    ax.set_xticklabels(gpqa_sizes)
    ax.set_xlim(1, 50)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(-5, 20)
    
    # Overall title
    # plt.suptitle('Model Improvements: Validation Loss and Performance Lifts',
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'combined_improvements_1x2.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


# Removed create_absolute_lift_1x2_plot - replaced by create_combined_improvement_1x2_plot




def create_performance_vs_dataset_size_plot(results: Dict):
    """Create a 1x2 plot showing performance improvement vs dataset size"""
    
    # Combine AIME scores first
    combined_results = combine_aime_scores(results)
    
    # Define dataset sizes and model sizes
    dataset_sizes = [29, 58, 116, 232, 464, 928]
    model_sizes = ['1.5B', '7B', '14B', '32B']
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Define colors for model sizes
    model_colors = {
        '1.5B': '#FF6B6B',
        '7B': '#4ECDC4', 
        '14B': '#2E86AB',
        '32B': '#A23B72'
    }
    
    # Markers for model sizes
    model_markers = {
        '1.5B': 'o',
        '7B': 's',
        '14B': '^',
        '32B': 'D'
    }
    
    # Plot 1: Combined AIME (left)
    ax = axes[0]
    
    for model_size in model_sizes:
        improvements = []
        errors = []
        valid_dataset_sizes = []
        
        # Get base and DeepSeek scores for this model size
        base_score = combined_results.get(model_size, {}).get('base', {}).get('aime_combined', {}).get('score', 0)
        deepseek_score = combined_results.get(model_size, {}).get('deepseek', {}).get('aime_combined', {}).get('score', 0)
        deepseek_improvement = (deepseek_score - base_score) * 100  # Convert to percentage points
        
        for dataset_size in dataset_sizes:
            variant = f'ft-{dataset_size}'
            if model_size in combined_results and variant in combined_results[model_size]:
                ft_data = combined_results[model_size][variant].get('aime_combined', {})
                if ft_data:
                    ft_score = ft_data.get('score', 0)
                    ft_stderr = ft_data.get('stderr', 0)
                    
                    improvement = (ft_score - base_score) * 100  # Convert to percentage points
                    improvements.append(improvement)
                    errors.append(ft_stderr * 100)
                    valid_dataset_sizes.append(dataset_size)
        
        if improvements:
            # Plot line with markers
            ax.errorbar(valid_dataset_sizes, improvements,
                       yerr=errors,
                       marker=model_markers[model_size],
                       label=f'{model_size}',
                       linewidth=2, markersize=8, 
                       color=model_colors[model_size],
                       capsize=5, capthick=1.5)
            
            # Add DeepSeek reference line (dotted)
            if deepseek_improvement != 0:
                ax.axhline(y=deepseek_improvement, 
                          color=model_colors[model_size], 
                          linestyle=':', 
                          alpha=0.5, 
                          linewidth=1.5)
    
    ax.set_xlabel('Training Dataset Size', fontsize=11)
    ax.set_ylabel('Performance Improvement (percentage points)', fontsize=11)
    ax.set_title('AIME 2024 & 2025', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels(dataset_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, title='Model Size')
    
    # Plot 2: GPQA (right)
    ax = axes[1]
    
    for model_size in model_sizes:
        improvements = []
        errors = []
        valid_dataset_sizes = []
        
        # Get base and DeepSeek scores for this model size
        base_score = results.get(model_size, {}).get('base', {}).get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
        deepseek_score = results.get(model_size, {}).get('deepseek', {}).get('gpqa_diamond_cot_zeroshot', {}).get('score', 0)
        deepseek_improvement = (deepseek_score - base_score) * 100  # Convert to percentage points
        
        for dataset_size in dataset_sizes:
            variant = f'ft-{dataset_size}'
            if model_size in results and variant in results[model_size]:
                ft_data = results[model_size][variant].get('gpqa_diamond_cot_zeroshot', {})
                if ft_data:
                    ft_score = ft_data.get('score', 0)
                    ft_stderr = ft_data.get('stderr', 0)
                    
                    improvement = (ft_score - base_score) * 100  # Convert to percentage points
                    improvements.append(improvement)
                    errors.append(ft_stderr * 100)
                    valid_dataset_sizes.append(dataset_size)
        
        if improvements:
            # Plot line with markers
            ax.errorbar(valid_dataset_sizes, improvements,
                       yerr=errors,
                       marker=model_markers[model_size],
                       label=f'{model_size}',
                       linewidth=2, markersize=8,
                       color=model_colors[model_size],
                       capsize=5, capthick=1.5)
            
            # Add DeepSeek reference line (dotted)
            if deepseek_improvement != 0:
                ax.axhline(y=deepseek_improvement,
                          color=model_colors[model_size],
                          linestyle=':',
                          alpha=0.5,
                          linewidth=1.5)
    
    ax.set_xlabel('Training Dataset Size', fontsize=11)
    ax.set_ylabel('Performance Improvement (percentage points)', fontsize=11)
    ax.set_title('GPQA Diamond', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels(dataset_sizes)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, title='Model Size')
    
    # Overall title
    # plt.suptitle('Performance Scaling with Training Dataset Size',
    #              fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plot_path = PLOTS_DIR / 'performance_vs_dataset_size.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {plot_path}")
    plt.show()


def main():
    """Main function to generate performance recovery plots"""
    print("\n" + "="*80)
    print("PERFORMANCE RECOVERY PLOTS")
    print("="*80 + "\n")

    # Parse results from directory
    print("Parsing results directory...")
    results = parse_results_directory(base_path="results/temperature")

    # Generate plots
    print("\nGenerating combined 1x2 plot...")
    create_combined_1x2_plot(results)

    print("\nGenerating recovery 1x2 plot...")
    create_recovery_1x2_plot(results)

    print("\nGenerating combined improvement plot...")
    create_combined_improvement_1x2_plot(results)

    print("\nGenerating performance vs dataset size plot...")
    create_performance_vs_dataset_size_plot(results)

    print("\nPerformance recovery plots completed!")


if __name__ == "__main__":
    main()
