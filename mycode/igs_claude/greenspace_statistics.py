"""
Comprehensive Greenspace Statistics Module for Remote Sensing Journal Publications

This module calculates advanced metrics including:
- Classification accuracy metrics (OA, Precision, Recall, F1, Kappa, IoU)
- Landscape metrics (patch analysis, fragmentation, connectivity)
- Spatial statistics (edge density, shape index, aggregation)
- Statistical summaries with confidence intervals
"""

import numpy as np
import csv
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import ndimage
from scipy.ndimage import label
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')


class GreenspaceStatistics:
    """Calculate comprehensive statistics for greenspace analysis"""

    def __init__(self, pixel_resolution_m: float = 10.0):
        """
        Initialize statistics calculator

        Args:
            pixel_resolution_m: Spatial resolution in meters per pixel (default 10m for zoom 18)
        """
        self.pixel_resolution = pixel_resolution_m
        self.pixel_area_m2 = pixel_resolution_m ** 2
        self.pixel_area_km2 = self.pixel_area_m2 / 1e6

    def calculate_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate confusion matrix based metrics

        Returns classification accuracy metrics suitable for remote sensing journals
        """
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        total = tn + fp + fn + tp
        overall_accuracy = (tp + tn) / total if total > 0 else 0

        # Producer's accuracy (Recall/Sensitivity) - for greenspace class
        producers_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0

        # User's accuracy (Precision) - for greenspace class
        users_accuracy = tp / (tp + fp) if (tp + fp) > 0 else 0

        # F1 Score
        f1_score = 2 * (users_accuracy * producers_accuracy) / \
                   (users_accuracy + producers_accuracy) if (users_accuracy + producers_accuracy) > 0 else 0

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true_flat, y_pred_flat)

        # Intersection over Union (IoU) / Jaccard Index
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'overall_accuracy': overall_accuracy * 100,
            'producers_accuracy_greenspace': producers_accuracy * 100,
            'users_accuracy_greenspace': users_accuracy * 100,
            'f1_score': f1_score,
            'cohens_kappa': kappa,
            'iou_jaccard': iou,
            'specificity': specificity * 100,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_pixels': int(total)
        }

    def calculate_patch_metrics(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate landscape patch metrics

        Analyzes spatial configuration and fragmentation of greenspace patches
        """
        # Label connected components
        labeled_array, num_patches = label(mask)

        if num_patches == 0:
            return {
                'num_patches': 0,
                'total_greenspace_pixels': 0,
                'total_greenspace_area_m2': 0,
                'total_greenspace_area_km2': 0,
                'mean_patch_size_m2': 0,
                'median_patch_size_m2': 0,
                'std_patch_size_m2': 0,
                'largest_patch_size_m2': 0,
                'smallest_patch_size_m2': 0,
                'patch_density_per_km2': 0,
                'mean_patch_perimeter_m': 0,
                'total_edge_length_m': 0
            }

        # Calculate patch sizes
        patch_sizes_pixels = []
        patch_perimeters = []

        for patch_id in range(1, num_patches + 1):
            patch_mask = (labeled_array == patch_id)
            patch_size = np.sum(patch_mask)
            patch_sizes_pixels.append(patch_size)

            # Calculate perimeter (edge pixels)
            eroded = ndimage.binary_erosion(patch_mask)
            edge = patch_mask & ~eroded
            perimeter_pixels = np.sum(edge)
            patch_perimeters.append(perimeter_pixels)

        patch_sizes_m2 = np.array(patch_sizes_pixels) * self.pixel_area_m2
        patch_perimeters_m = np.array(patch_perimeters) * self.pixel_resolution

        total_greenspace_pixels = np.sum(mask)
        total_greenspace_m2 = total_greenspace_pixels * self.pixel_area_m2
        total_greenspace_km2 = total_greenspace_m2 / 1e6

        # Calculate patch density (patches per km²)
        total_area_km2 = (mask.size * self.pixel_area_m2) / 1e6
        patch_density = num_patches / total_area_km2 if total_area_km2 > 0 else 0

        # Total edge length
        total_edge_length = np.sum(patch_perimeters_m)

        return {
            'num_patches': num_patches,
            'total_greenspace_pixels': int(total_greenspace_pixels),
            'total_greenspace_area_m2': float(total_greenspace_m2),
            'total_greenspace_area_km2': float(total_greenspace_km2),
            'mean_patch_size_m2': float(np.mean(patch_sizes_m2)),
            'median_patch_size_m2': float(np.median(patch_sizes_m2)),
            'std_patch_size_m2': float(np.std(patch_sizes_m2)),
            'largest_patch_size_m2': float(np.max(patch_sizes_m2)),
            'smallest_patch_size_m2': float(np.min(patch_sizes_m2)),
            'patch_density_per_km2': float(patch_density),
            'mean_patch_perimeter_m': float(np.mean(patch_perimeters_m)),
            'total_edge_length_m': float(total_edge_length),
            'edge_density_m_per_ha': float(total_edge_length / (total_area_km2 * 100)) if total_area_km2 > 0 else 0
        }

    def calculate_landscape_metrics(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate advanced landscape-level metrics

        Includes fragmentation, shape complexity, and aggregation indices
        """
        labeled_array, num_patches = label(mask)

        if num_patches == 0:
            return {
                'landscape_shape_index': 0,
                'aggregation_index': 0,
                'fragmentation_index': 0,
                'largest_patch_index': 0,
                'effective_mesh_size_km2': 0
            }

        # Landscape Shape Index (LSI)
        # Measures shape complexity - higher values = more complex/fragmented
        total_edge = 0
        total_area = np.sum(mask) * self.pixel_area_m2

        for patch_id in range(1, num_patches + 1):
            patch_mask = (labeled_array == patch_id)
            eroded = ndimage.binary_erosion(patch_mask)
            edge = patch_mask & ~eroded
            total_edge += np.sum(edge) * self.pixel_resolution

        # LSI = total edge / minimum edge (edge of a circle with same area)
        min_edge = 2 * np.sqrt(np.pi * total_area) if total_area > 0 else 1
        lsi = total_edge / min_edge if min_edge > 0 else 0

        # Aggregation Index
        # Measures clustering of greenspace (0-100, higher = more aggregated)
        patch_sizes = []
        for patch_id in range(1, num_patches + 1):
            patch_mask = (labeled_array == patch_id)
            patch_sizes.append(np.sum(patch_mask) * self.pixel_area_m2)

        max_patch_size = max(patch_sizes)
        aggregation_index = (max_patch_size / total_area * 100) if total_area > 0 else 0

        # Fragmentation Index (inverse of mean patch size)
        mean_patch_size = total_area / num_patches if num_patches > 0 else 0
        fragmentation_index = 1 / (mean_patch_size / 1000) if mean_patch_size > 0 else 0

        # Largest Patch Index (%)
        largest_patch_index = (max_patch_size / total_area * 100) if total_area > 0 else 0

        # Effective Mesh Size (MESH)
        # Average patch size weighted by patch area
        mesh_size = sum([(size ** 2) for size in patch_sizes]) / total_area if total_area > 0 else 0
        mesh_size_km2 = mesh_size / 1e6

        return {
            'landscape_shape_index': float(lsi),
            'aggregation_index': float(aggregation_index),
            'fragmentation_index': float(fragmentation_index),
            'largest_patch_index': float(largest_patch_index),
            'effective_mesh_size_km2': float(mesh_size_km2)
        }

    def calculate_coverage_statistics(
        self,
        masks: List[np.ndarray],
        calculate_ci: bool = True
    ) -> Dict[str, float]:
        """
        Calculate statistical summaries of greenspace coverage

        Args:
            masks: List of prediction masks
            calculate_ci: Whether to calculate 95% confidence intervals
        """
        coverage_percentages = []

        for mask in masks:
            total_pixels = mask.size
            greenspace_pixels = np.sum(mask > 0)
            coverage = (greenspace_pixels / total_pixels) * 100
            coverage_percentages.append(coverage)

        coverage_array = np.array(coverage_percentages)

        stats = {
            'mean_coverage_percent': float(np.mean(coverage_array)),
            'median_coverage_percent': float(np.median(coverage_array)),
            'std_coverage_percent': float(np.std(coverage_array)),
            'min_coverage_percent': float(np.min(coverage_array)),
            'max_coverage_percent': float(np.max(coverage_array)),
            'range_coverage_percent': float(np.max(coverage_array) - np.min(coverage_array)),
            'coefficient_of_variation': float(np.std(coverage_array) / np.mean(coverage_array)) if np.mean(coverage_array) > 0 else 0,
            'q1_coverage_percent': float(np.percentile(coverage_array, 25)),
            'q3_coverage_percent': float(np.percentile(coverage_array, 75)),
            'iqr_coverage_percent': float(np.percentile(coverage_array, 75) - np.percentile(coverage_array, 25))
        }

        # Calculate 95% confidence interval
        if calculate_ci and len(coverage_array) > 1:
            sem = np.std(coverage_array, ddof=1) / np.sqrt(len(coverage_array))
            ci_95 = 1.96 * sem  # Assuming normal distribution
            stats['ci_95_lower'] = float(np.mean(coverage_array) - ci_95)
            stats['ci_95_upper'] = float(np.mean(coverage_array) + ci_95)

        return stats

    def calculate_all_metrics(
        self,
        predictions: List[np.ndarray],
        ground_truth: Optional[List[np.ndarray]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Calculate all metrics for comprehensive analysis

        Args:
            predictions: List of predicted masks
            ground_truth: Optional list of ground truth masks for accuracy assessment
            bbox: Optional bounding box (west, south, east, north)
            metadata: Optional metadata dictionary

        Returns:
            Dictionary containing all calculated metrics
        """
        # Merge all predictions into single mask for landscape analysis
        merged_predictions = np.concatenate([mask.flatten() for mask in predictions])
        merged_predictions = merged_predictions.reshape(-1, predictions[0].shape[0])

        # Calculate metrics
        all_metrics = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_tiles': len(predictions),
                'tile_size': predictions[0].shape,
                'pixel_resolution_m': self.pixel_resolution,
                'bbox': bbox if bbox else 'Not provided'
            }
        }

        # Add custom metadata
        if metadata:
            all_metrics['metadata'].update(metadata)

        # Coverage statistics
        all_metrics['coverage_statistics'] = self.calculate_coverage_statistics(predictions)

        # Patch metrics (using merged mask)
        all_metrics['patch_metrics'] = self.calculate_patch_metrics(merged_predictions > 0)

        # Landscape metrics
        all_metrics['landscape_metrics'] = self.calculate_landscape_metrics(merged_predictions > 0)

        # Accuracy metrics (if ground truth available)
        if ground_truth is not None:
            merged_ground_truth = np.concatenate([mask.flatten() for mask in ground_truth])
            merged_ground_truth = merged_ground_truth.reshape(-1, ground_truth[0].shape[0])
            all_metrics['accuracy_metrics'] = self.calculate_confusion_matrix_metrics(
                merged_ground_truth > 0,
                merged_predictions > 0
            )

        return all_metrics


def export_metrics_to_csv(
    metrics: Dict[str, any],
    output_path: str = "output/greenspace_statistics.csv",
    append_mode: bool = False
):
    """
    Export metrics to CSV file suitable for journal publications

    Args:
        metrics: Dictionary of metrics from calculate_all_metrics()
        output_path: Path to output CSV file
        append_mode: If True, append to existing file
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Flatten nested dictionary
    flattened = {}

    for category, values in metrics.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flattened[f"{category}_{key}"] = value
        else:
            flattened[category] = values

    # Write to CSV
    mode = 'a' if append_mode and os.path.exists(output_path) else 'w'
    file_exists = os.path.exists(output_path) and append_mode

    with open(output_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=flattened.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(flattened)

    print(f"\nStatistics exported to: {output_path}")
    return output_path


def export_summary_report(
    metrics: Dict[str, any],
    output_path: str = "output/greenspace_report.txt"
):
    """
    Export human-readable summary report

    Creates a formatted text report suitable for interpretation
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GREENSPACE ANALYSIS REPORT\n")
        f.write("Comprehensive Statistics for Remote Sensing Publication\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        f.write("METADATA\n")
        f.write("-" * 80 + "\n")
        for key, value in metrics.get('metadata', {}).items():
            f.write(f"{key:30s}: {value}\n")
        f.write("\n")

        # Coverage Statistics
        if 'coverage_statistics' in metrics:
            f.write("COVERAGE STATISTICS\n")
            f.write("-" * 80 + "\n")
            cov = metrics['coverage_statistics']
            f.write(f"{'Mean Coverage':30s}: {cov.get('mean_coverage_percent', 0):.2f}%\n")
            f.write(f"{'Median Coverage':30s}: {cov.get('median_coverage_percent', 0):.2f}%\n")
            f.write(f"{'Std Deviation':30s}: {cov.get('std_coverage_percent', 0):.2f}%\n")
            f.write(f"{'Range':30s}: {cov.get('min_coverage_percent', 0):.2f}% - {cov.get('max_coverage_percent', 0):.2f}%\n")
            if 'ci_95_lower' in cov:
                f.write(f"{'95% Confidence Interval':30s}: [{cov['ci_95_lower']:.2f}%, {cov['ci_95_upper']:.2f}%]\n")
            f.write("\n")

        # Patch Metrics
        if 'patch_metrics' in metrics:
            f.write("PATCH METRICS\n")
            f.write("-" * 80 + "\n")
            patch = metrics['patch_metrics']
            f.write(f"{'Number of Patches':30s}: {patch.get('num_patches', 0)}\n")
            f.write(f"{'Total Greenspace Area':30s}: {patch.get('total_greenspace_area_km2', 0):.4f} km²\n")
            f.write(f"{'Mean Patch Size':30s}: {patch.get('mean_patch_size_m2', 0):.2f} m²\n")
            f.write(f"{'Largest Patch':30s}: {patch.get('largest_patch_size_m2', 0):.2f} m²\n")
            f.write(f"{'Patch Density':30s}: {patch.get('patch_density_per_km2', 0):.2f} patches/km²\n")
            f.write(f"{'Edge Density':30s}: {patch.get('edge_density_m_per_ha', 0):.2f} m/ha\n")
            f.write("\n")

        # Landscape Metrics
        if 'landscape_metrics' in metrics:
            f.write("LANDSCAPE METRICS\n")
            f.write("-" * 80 + "\n")
            land = metrics['landscape_metrics']
            f.write(f"{'Landscape Shape Index':30s}: {land.get('landscape_shape_index', 0):.3f}\n")
            f.write(f"{'Aggregation Index':30s}: {land.get('aggregation_index', 0):.2f}%\n")
            f.write(f"{'Fragmentation Index':30s}: {land.get('fragmentation_index', 0):.4f}\n")
            f.write(f"{'Largest Patch Index':30s}: {land.get('largest_patch_index', 0):.2f}%\n")
            f.write(f"{'Effective Mesh Size':30s}: {land.get('effective_mesh_size_km2', 0):.6f} km²\n")
            f.write("\n")

        # Accuracy Metrics
        if 'accuracy_metrics' in metrics:
            f.write("ACCURACY ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            acc = metrics['accuracy_metrics']
            f.write(f"{'Overall Accuracy':30s}: {acc.get('overall_accuracy', 0):.2f}%\n")
            prod_acc_label = "Producer's Accuracy"
            f.write(f"{prod_acc_label:30s}: {acc.get('producers_accuracy_greenspace', 0):.2f}%\n")
            user_acc_label = "User's Accuracy"
            f.write(f"{user_acc_label:30s}: {acc.get('users_accuracy_greenspace', 0):.2f}%\n")
            f.write(f"{'F1 Score':30s}: {acc.get('f1_score', 0):.4f}\n")
            kappa_label = "Cohen's Kappa"
            f.write(f"{kappa_label:30s}: {acc.get('cohens_kappa', 0):.4f}\n")
            f.write(f"{'IoU (Jaccard Index)':30s}: {acc.get('iou_jaccard', 0):.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"  True Positives:  {acc.get('true_positives', 0):,}\n")
            f.write(f"  True Negatives:  {acc.get('true_negatives', 0):,}\n")
            f.write(f"  False Positives: {acc.get('false_positives', 0):,}\n")
            f.write(f"  False Negatives: {acc.get('false_negatives', 0):,}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Report generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 80 + "\n")

    print(f"Summary report exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Greenspace Statistics Module")
    print("=" * 80)
    print("\nThis module calculates comprehensive statistics for remote sensing analysis.")
    print("\nKey features:")
    print("  - Classification accuracy metrics (OA, Kappa, IoU, F1)")
    print("  - Landscape metrics (fragmentation, shape index, aggregation)")
    print("  - Patch analysis (size distribution, density, connectivity)")
    print("  - Statistical summaries with confidence intervals")
    print("  - CSV export for journal publications")
    print("\nExample usage:")
    print("  from greenspace_statistics import GreenspaceStatistics, export_metrics_to_csv")
    print("  ")
    print("  # Initialize calculator")
    print("  stats = GreenspaceStatistics(pixel_resolution_m=10.0)")
    print("  ")
    print("  # Calculate all metrics")
    print("  metrics = stats.calculate_all_metrics(predictions, ground_truth, bbox)")
    print("  ")
    print("  # Export to CSV")
    print("  export_metrics_to_csv(metrics, 'output/greenspace_statistics.csv')")
