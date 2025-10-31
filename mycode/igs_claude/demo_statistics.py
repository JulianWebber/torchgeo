"""
Demo script to generate statistics from synthetic or small sample data

This allows testing the statistics module without running full analysis
"""

import numpy as np
import os
from greenspace_statistics import (
    GreenspaceStatistics,
    export_metrics_to_csv,
    export_summary_report
)


def generate_synthetic_data(num_tiles=10, tile_size=256):
    """
    Generate synthetic greenspace predictions for testing

    Creates realistic-looking patterns with varying coverage
    """
    from scipy.ndimage import gaussian_filter

    predictions = []
    ground_truth = []

    for i in range(num_tiles):
        # Generate random greenspace pattern
        random_field = np.random.rand(tile_size, tile_size)
        smoothed = gaussian_filter(random_field, sigma=20)

        # Vary threshold to get different coverage levels
        threshold = 0.3 + 0.3 * np.random.rand()
        pred_mask = (smoothed > threshold).astype(np.uint8)

        # Create similar but slightly different ground truth
        noise = np.random.rand(tile_size, tile_size) * 0.1
        gt_smoothed = smoothed + noise
        gt_mask = (gt_smoothed > threshold).astype(np.uint8)

        predictions.append(pred_mask)
        ground_truth.append(gt_mask)

    return predictions, ground_truth


def main():
    """Run statistics demo"""
    print("=" * 80)
    print("GREENSPACE STATISTICS - DEMO")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic test data...")
    num_tiles = 20
    tile_size = 256
    predictions, ground_truth = generate_synthetic_data(num_tiles, tile_size)
    print(f"  Created {num_tiles} tiles of {tile_size}x{tile_size} pixels")
    print()

    # Initialize statistics calculator
    print("Calculating comprehensive statistics...")
    stats = GreenspaceStatistics(pixel_resolution_m=10.0)

    # Define example bbox
    bbox = (139.615, 35.461, 139.628, 35.471)

    # Metadata
    metadata = {
        'data_type': 'synthetic',
        'area_name': 'Demo Area',
        'model': 'U-Net',
        'purpose': 'Testing statistics module'
    }

    # Calculate all metrics
    metrics = stats.calculate_all_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        bbox=bbox,
        metadata=metadata
    )

    print("Statistics calculation complete!")
    print()

    # Export results
    print("Exporting results...")
    csv_path = export_metrics_to_csv(metrics, "output/demo_statistics.csv")
    report_path = export_summary_report(metrics, "output/demo_report.txt")
    print()

    # Display summary
    print("=" * 80)
    print("DEMO RESULTS SUMMARY")
    print("=" * 80)
    print()

    if 'coverage_statistics' in metrics:
        cov = metrics['coverage_statistics']
        print("COVERAGE STATISTICS:")
        print(f"  Mean Coverage:       {cov['mean_coverage_percent']:.2f}%")
        print(f"  Std Deviation:       {cov['std_coverage_percent']:.2f}%")
        print(f"  Range:               {cov['min_coverage_percent']:.2f}% - {cov['max_coverage_percent']:.2f}%")
        if 'ci_95_lower' in cov:
            print(f"  95% CI:              [{cov['ci_95_lower']:.2f}%, {cov['ci_95_upper']:.2f}%]")
        print()

    if 'patch_metrics' in metrics:
        patch = metrics['patch_metrics']
        print("PATCH METRICS:")
        print(f"  Number of Patches:   {patch['num_patches']}")
        print(f"  Total Area:          {patch['total_greenspace_area_km2']:.4f} km²")
        print(f"  Mean Patch Size:     {patch['mean_patch_size_m2']:.2f} m²")
        print(f"  Largest Patch:       {patch['largest_patch_size_m2']:.2f} m²")
        print(f"  Patch Density:       {patch['patch_density_per_km2']:.2f} patches/km²")
        print(f"  Edge Density:        {patch['edge_density_m_per_ha']:.2f} m/ha")
        print()

    if 'landscape_metrics' in metrics:
        land = metrics['landscape_metrics']
        print("LANDSCAPE METRICS:")
        print(f"  Shape Index:         {land['landscape_shape_index']:.3f}")
        print(f"  Aggregation Index:   {land['aggregation_index']:.2f}%")
        print(f"  Fragmentation Index: {land['fragmentation_index']:.4f}")
        print(f"  Largest Patch Index: {land['largest_patch_index']:.2f}%")
        print(f"  Effective Mesh Size: {land['effective_mesh_size_km2']:.6f} km²")
        print()

    if 'accuracy_metrics' in metrics:
        acc = metrics['accuracy_metrics']
        print("ACCURACY METRICS:")
        print(f"  Overall Accuracy:    {acc['overall_accuracy']:.2f}%")
        print(f"  Producer's Accuracy: {acc['producers_accuracy_greenspace']:.2f}%")
        print(f"  User's Accuracy:     {acc['users_accuracy_greenspace']:.2f}%")
        print(f"  F1 Score:            {acc['f1_score']:.4f}")
        print(f"  Cohen's Kappa:       {acc['cohens_kappa']:.4f}")
        print(f"  IoU (Jaccard):       {acc['iou_jaccard']:.4f}")
        print()
        print("  Confusion Matrix:")
        print(f"    True Positives:    {acc['true_positives']:,}")
        print(f"    True Negatives:    {acc['true_negatives']:,}")
        print(f"    False Positives:   {acc['false_positives']:,}")
        print(f"    False Negatives:   {acc['false_negatives']:,}")
        print()

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    print()
    print("These files are suitable for submission to remote sensing journals.")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
