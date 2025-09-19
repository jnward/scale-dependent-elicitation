#!/usr/bin/env python3
"""
Shared data parsing and utility functions for evaluation analysis
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Create plots directory if it doesn't exist
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def get_variant_label(variant: str) -> str:
    """Convert variant name to a human-readable label."""
    if variant == 'base':
        return 'Qwen2.5-Instruct (baseline)'
    elif variant == 'deepseek':
        return 'Deepseek-R1-Distill-Qwen (skyline)'
    elif variant == 'lora':
        return 'LoRA finetune'
    elif variant.startswith('ft-'):
        num_samples = variant.replace('ft-', '')
        return f'Full finetune ({num_samples} examples)'
    else:
        return variant


def parse_aime_multiple_runs(size: str, variant: str) -> Tuple[float, float, float, float]:
    """
    Parse AIME results from multiple runs and return averaged scores with standard errors.

    Args:
        size: Model size (e.g., "32B", "14B")
        variant: Model variant (e.g., "base", "ft-928", "deepseek")

    Returns:
        Tuple of (aime24_mean, aime24_stderr, aime25_mean, aime25_stderr)
    """
    # Directories containing AIME runs - dynamically check what exists
    aime_dirs = [
        "results/temperature",
        "results/temperature_aime_2",
        "results/temperature_aime_3",
        "results/temperature_aime_4"
    ]

    # Filter to only existing directories
    aime_dirs = [d for d in aime_dirs if Path(d).exists()]

    # Construct directory name based on size and variant
    if variant == "base":
        dir_name = f"{size}-base"
    elif variant == "deepseek":
        dir_name = f"{size}-deepseek"
    else:  # ft-464, ft-928, ft-232, ft-116
        dir_name = f"{size}-{variant}"

    aime24_scores = []
    aime25_scores = []

    for base_dir in aime_dirs:
        model_path = Path(base_dir) / dir_name
        if model_path.exists() and model_path.is_dir():
            # Find results files
            for subdir in model_path.iterdir():
                if subdir.is_dir():
                    json_files = list(subdir.glob("results_*.json"))
                    if json_files:
                        json_file = sorted(json_files)[-1]  # Use most recent

                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)

                            if "results" in data:
                                # Check for AIME24
                                if "aime24" in data["results"]:
                                    score = data["results"]["aime24"].get("exact_match,none", None)
                                    if score is not None:
                                        aime24_scores.append(score)

                                # Check for AIME25
                                if "aime25" in data["results"]:
                                    score = data["results"]["aime25"].get("exact_match,none", None)
                                    if score is not None:
                                        aime25_scores.append(score)

                                # Break if we found at least one AIME result
                                if aime24_scores or aime25_scores:
                                    break
                        except Exception as e:
                            print(f"Warning: Could not read {json_file}: {e}")

    # Calculate mean and standard error for AIME24
    if aime24_scores:
        aime24_mean = np.mean(aime24_scores)
        if len(aime24_scores) > 1:
            aime24_std = np.std(aime24_scores, ddof=1)
            aime24_stderr = aime24_std / np.sqrt(len(aime24_scores))
        else:
            aime24_stderr = 0.0
    else:
        aime24_mean = 0.0
        aime24_stderr = 0.0

    # Calculate mean and standard error for AIME25
    if aime25_scores:
        aime25_mean = np.mean(aime25_scores)
        if len(aime25_scores) > 1:
            aime25_std = np.std(aime25_scores, ddof=1)
            aime25_stderr = aime25_std / np.sqrt(len(aime25_scores))
        else:
            aime25_stderr = 0.0
    else:
        aime25_mean = 0.0
        aime25_stderr = 0.0

    return aime24_mean, aime24_stderr, aime25_mean, aime25_stderr


def parse_gpqa_multiple_runs(size: str, variant: str) -> Tuple[float, float]:
    """
    Parse GPQA results from multiple runs and return averaged score with standard error.

    Args:
        size: Model size (e.g., "32B", "14B")
        variant: Model variant (e.g., "base", "ft-928", "deepseek")

    Returns:
        Tuple of (mean_score, standard_error)
    """
    # Directories containing GPQA runs
    gpqa_dirs = [
        "results/temperature",
        "results/temperature_gpqa_2",
        "results/temperature_gpqa_3",
        "results/temperature_gpqa_4"
    ]

    # Construct directory name based on size and variant
    if variant == "base":
        dir_name = f"{size}-base"
    elif variant == "deepseek":
        dir_name = f"{size}-deepseek"
    else:  # ft-464, ft-928, ft-232, ft-116
        dir_name = f"{size}-{variant}"

    scores = []

    for base_dir in gpqa_dirs:
        model_path = Path(base_dir) / dir_name
        if model_path.exists() and model_path.is_dir():
            # Find results files
            for subdir in model_path.iterdir():
                if subdir.is_dir():
                    json_files = list(subdir.glob("results_*.json"))
                    if json_files:
                        json_file = sorted(json_files)[-1]  # Use most recent

                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)

                            if "results" in data and "gpqa_diamond_cot_zeroshot" in data["results"]:
                                score = data["results"]["gpqa_diamond_cot_zeroshot"].get("exact_match,none", None)
                                if score is not None:
                                    scores.append(score)
                                    break  # Found GPQA results in this directory
                        except Exception as e:
                            print(f"Warning: Could not read {json_file}: {e}")

    if not scores:
        return 0.0, 0.0

    # Calculate mean and standard error
    mean_score = np.mean(scores)
    if len(scores) > 1:
        std_score = np.std(scores, ddof=1)  # Sample standard deviation
        stderr = std_score / np.sqrt(len(scores))
    else:
        stderr = 0.0

    return mean_score, stderr


def parse_results_directory(base_path: str = "results/temperature") -> Dict:
    """
    Parse all results from the results directory structure.
    Returns nested dictionary with structure:
    {size: {variant: {benchmark: {score, stderr}}}}
    """
    results = {}

    # Get all unique size-variant combinations from first temperature directory
    # This gives us the structure
    for size_dir in sorted(Path(base_path).iterdir()):
        if not size_dir.is_dir():
            continue

        # Extract size (e.g., "1.5B", "7B", etc.)
        size_name = size_dir.name

        # Parse size for sorting
        if "-base" in size_name:
            size = size_name.replace("-base", "")
            variant = "base"
        elif "-deepseek" in size_name:
            size = size_name.replace("-deepseek", "")
            variant = "deepseek"
        elif "-lora" in size_name:
            size = size_name.replace("-lora", "")
            variant = "lora"
        elif "-ft-" in size_name:
            parts = size_name.split("-ft-")
            size = parts[0]
            variant = f"ft-{parts[1]}"
        else:
            continue

        # Initialize nested structure
        if size not in results:
            results[size] = {}
        if variant not in results[size]:
            results[size][variant] = {}

    # Now handle AIME with multi-run averaging
    for size in results.keys():
        for variant in results[size].keys():
            aime24_mean, aime24_stderr, aime25_mean, aime25_stderr = parse_aime_multiple_runs(size, variant)

            # Add AIME24 results if found
            if aime24_mean > 0 or aime24_stderr > 0:
                results[size][variant]["aime24"] = {
                    "score": aime24_mean,
                    "stderr": aime24_stderr
                }

            # Add AIME25 results if found
            if aime25_mean > 0 or aime25_stderr > 0:
                results[size][variant]["aime25"] = {
                    "score": aime25_mean,
                    "stderr": aime25_stderr
                }

    # Handle GPQA separately with multi-run averaging
    for size in results.keys():
        for variant in results[size].keys():
            gpqa_mean, gpqa_stderr = parse_gpqa_multiple_runs(size, variant)
            if gpqa_mean > 0 or gpqa_stderr > 0:  # Only add if we found GPQA results
                results[size][variant]["gpqa_diamond_cot_zeroshot"] = {
                    "score": gpqa_mean,
                    "stderr": gpqa_stderr
                }

    return results


def combine_aime_scores(results: Dict) -> Dict:
    """
    Combine AIME24 and AIME25 scores into a single 'aime_combined' metric.
    Now uses proper standard error from multiple runs instead of bootstrapping.
    """
    combined_results = {}

    for size in results.keys():
        combined_results[size] = {}
        for variant in results[size].keys():
            # Collect individual scores
            aime24_data = results[size][variant].get('aime24', {})
            aime25_data = results[size][variant].get('aime25', {})

            # Get the scores and standard errors
            aime24_score = aime24_data.get('score', 0.0)
            aime24_stderr = aime24_data.get('stderr', 0.0)
            aime25_score = aime25_data.get('score', 0.0)
            aime25_stderr = aime25_data.get('stderr', 0.0)

            # For combined score, average the two scores
            # This is equivalent to treating it as 60 total questions
            combined_score = (aime24_score + aime25_score) / 2

            # For combined standard error, combine the errors
            # Using error propagation for the average of two values
            # SE(avg) = sqrt((SE1^2 + SE2^2) / 4)
            # But since we want the SE of the combined score (not average),
            # and both tests have same number of questions,
            # we can use: SE_combined = sqrt(SE1^2 + SE2^2) / 2
            if aime24_stderr > 0 or aime25_stderr > 0:
                combined_stderr = np.sqrt(aime24_stderr**2 + aime25_stderr**2) / 2
            else:
                combined_stderr = 0.0

            combined_results[size][variant] = {
                'aime_combined': {
                    'score': combined_score,
                    'stderr': combined_stderr
                },
                'aime24': aime24_data,
                'aime25': aime25_data
            }

    # Also copy over any other data that's not AIME-related
    for size in results.keys():
        for variant in results[size].keys():
            for key in results[size][variant].keys():
                if key not in ['aime24', 'aime25'] and key not in combined_results[size][variant]:
                    combined_results[size][variant][key] = results[size][variant][key]

    return combined_results


def calculate_performance_recovered(ft_score: float, base_score: float, deepseek_score: float) -> Tuple[float, bool]:
    """
    Calculate percentage of performance gap recovered.
    Returns (percentage, is_valid) where is_valid indicates if the calculation is meaningful.
    """
    gap = deepseek_score - base_score
    if gap <= 0:
        return 0.0, False
    recovered = (ft_score - base_score) / gap * 100
    return recovered, True


def save_parsed_data(results: Dict, filename: str = "new_evaluation_data.json"):
    """Save parsed results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved parsed data to {filename}")


def create_summary_table(results: Dict, filename: str = "new_evaluation_summary.csv"):
    """Create and save a summary CSV table"""
    rows = []

    # Header row
    header = ['Model Size', 'Variant', 'AIME24', 'AIME25', 'AIME Combined', 'GPQA']

    # Data rows
    for size in ['32B', '14B', '7B', '1.5B']:
        if size not in results:
            continue
        for variant in ['base', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7', 'deepseek', 'lora']:
            if variant not in results[size]:
                continue

            row = [
                size,
                get_variant_label(variant),
                f"{results[size][variant].get('aime24', {}).get('score', 0):.1%}",
                f"{results[size][variant].get('aime25', {}).get('score', 0):.1%}",
                f"{results[size][variant].get('aime_combined', {}).get('score', 0):.1%}",
                f"{results[size][variant].get('gpqa', {}).get('score', 0):.1%}"
            ]
            rows.append(row)

    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved summary table to {filename}")

    # Print table to console
    print("\nSummary Table:")
    print("-" * 100)
    print(f"{'Model Size':<12} {'Variant':<35} {'AIME24':>8} {'AIME25':>8} {'Combined':>10} {'GPQA':>8}")
    print("-" * 100)
    for row in rows:
        print(f"{row[0]:<12} {row[1]:<35} {row[2]:>8} {row[3]:>8} {row[4]:>10} {row[5]:>8}")


def export_comprehensive_csv(results: Dict):
    """Export comprehensive results with all benchmarks and error bars to CSV"""
    # Prepare comprehensive CSV with all metrics
    csv_path = Path('comprehensive_results.csv')

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Variant', 'AIME24', 'AIME24_SE', 'AIME25', 'AIME25_SE',
                      'AIME_Combined', 'AIME_Combined_SE', 'GPQA', 'GPQA_SE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for size in ['32B', '14B', '7B', '1.5B']:
            if size not in results:
                continue
            for variant in ['base', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7', 'deepseek', 'lora']:
                if variant not in results[size]:
                    continue

                row = {
                    'Model': size,
                    'Variant': variant,
                    'AIME24': f"{results[size][variant].get('aime24', {}).get('score', 0):.3f}",
                    'AIME24_SE': f"{results[size][variant].get('aime24', {}).get('stderr', 0):.4f}",
                    'AIME25': f"{results[size][variant].get('aime25', {}).get('score', 0):.3f}",
                    'AIME25_SE': f"{results[size][variant].get('aime25', {}).get('stderr', 0):.4f}",
                    'AIME_Combined': f"{results[size][variant].get('aime_combined', {}).get('score', 0):.3f}",
                    'AIME_Combined_SE': f"{results[size][variant].get('aime_combined', {}).get('stderr', 0):.4f}",
                    'GPQA': f"{results[size][variant].get('gpqa', {}).get('score', 0):.3f}",
                    'GPQA_SE': f"{results[size][variant].get('gpqa', {}).get('stderr', 0):.4f}"
                }
                writer.writerow(row)

    print(f"Exported comprehensive results to {csv_path}")

    # Also create a simplified version for the paper (just combined AIME and GPQA)
    simple_csv_path = Path('results_for_paper.csv')

    with open(simple_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Variant', 'AIME_Combined', 'AIME_SE', 'GPQA', 'GPQA_SE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for size in ['32B', '14B', '7B', '1.5B']:
            if size not in results:
                continue
            for variant in ['base', 'ft-928', 'ft-464', 'ft-232', 'ft-116', 'ft-58', 'ft-29', 'ft-14', 'ft-7', 'deepseek', 'lora']:
                if variant not in results[size]:
                    continue

                row = {
                    'Model': size,
                    'Variant': variant,
                    'AIME_Combined': f"{results[size][variant].get('aime_combined', {}).get('score', 0)*100:.1f}",
                    'AIME_SE': f"{results[size][variant].get('aime_combined', {}).get('stderr', 0)*100:.2f}",
                    'GPQA': f"{results[size][variant].get('gpqa', {}).get('score', 0)*100:.1f}",
                    'GPQA_SE': f"{results[size][variant].get('gpqa', {}).get('stderr', 0)*100:.2f}"
                }
                writer.writerow(row)

    print(f"Exported simplified results for paper to {simple_csv_path}")