"""
Run greenspace extraction with comprehensive statistics export

This script integrates the greenspace extraction pipeline with advanced
statistical analysis suitable for remote sensing journal publications.
"""

import os
import sys
import torch
import numpy as np
from greenspace_simple import (
    get_tiles_for_bbox,
    train_model,
    predict_greenspace,
    GSITileDataset
)
from greenspace_statistics import (
    GreenspaceStatistics,
    export_metrics_to_csv,
    export_summary_report
)
from config import AREAS, TRAINING_CONFIG


def main():
    """Main execution with comprehensive statistics"""
    print("=" * 80)
    print("GREENSPACE EXTRACTION WITH COMPREHENSIVE STATISTICS")
    print("=" * 80)
    print()

    # Configuration
    # You can change this to any area from config.AREAS or define custom bbox
    area_name = 'yokohama_station'
    bbox = AREAS[area_name]

    # Or use custom bbox: (west, south, east, north)
    # bbox = (139.615, 35.461, 139.628, 35.471)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Configuration:")
    print(f"  Area: {area_name}")
    print(f"  Bounding Box: {bbox}")
    print(f"  Device: {device}")
    print(f"  Pixel Resolution: 10m (zoom level 18)")
    print()

    # Step 1: Download tiles
    print("Step 1: Downloading tiles...")
    print("-" * 80)
    tiles = get_tiles_for_bbox(bbox, zoom=18)

    if not tiles:
        print("ERROR: No tiles downloaded. Please check your internet connection.")
        return

    print(f"Successfully downloaded {len(tiles)} tiles")
    print()

    # Step 2: Train model
    print("Step 2: Training U-Net model...")
    print("-" * 80)
    epochs = TRAINING_CONFIG.get('epochs', 10)
    model = train_model(tiles, epochs=epochs, device=device)
    print()

    # Step 3: Generate predictions
    print("Step 3: Generating predictions...")
    print("-" * 80)
    predictions = []
    pseudo_labels = []  # For comparison with pseudo-labels

    dataset = GSITileDataset(tiles)

    for i, tile in enumerate(tiles):
        # Model prediction
        pred_mask = predict_greenspace(model, tile, device)
        predictions.append(pred_mask)

        # Pseudo-label for comparison
        pseudo_mask = dataset.create_vegetation_mask(tile)
        pseudo_labels.append(pseudo_mask)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(tiles)} tiles")

    print(f"Generated predictions for all {len(tiles)} tiles")
    print()

    # Step 4: Calculate comprehensive statistics
    print("Step 4: Calculating comprehensive statistics...")
    print("-" * 80)

    # Initialize statistics calculator (10m resolution for zoom 18)
    stats_calculator = GreenspaceStatistics(pixel_resolution_m=10.0)

    # Metadata
    metadata = {
        'area_name': area_name,
        'model_architecture': 'U-Net',
        'training_epochs': epochs,
        'device': device,
        'zoom_level': 18,
        'tile_size': '256x256',
        'pseudo_label_method': 'ExG+VARI+GLI combined indices'
    }

    # Calculate all metrics
    # Note: Using pseudo_labels as ground truth for demonstration
    # In real publication, use manually annotated ground truth
    metrics = stats_calculator.calculate_all_metrics(
        predictions=predictions,
        ground_truth=pseudo_labels,  # Using pseudo-labels as reference
        bbox=bbox,
        metadata=metadata
    )

    print("Statistics calculation complete!")
    print()

    # Step 5: Export results
    print("Step 5: Exporting results...")
    print("-" * 80)

    # Export to CSV
    csv_path = "output/greenspace_statistics.csv"
    export_metrics_to_csv(metrics, csv_path)

    # Export summary report
    report_path = "output/greenspace_report.txt"
    export_summary_report(metrics, report_path)

    # Save model
    model_path = "output/greenspace_unet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print()

    # Step 6: Print key results
    print("=" * 80)
    print("KEY RESULTS SUMMARY")
    print("=" * 80)
    print()

    if 'coverage_statistics' in metrics:
        cov = metrics['coverage_statistics']
        print("COVERAGE STATISTICS:")
        print(f"  Mean Coverage:     {cov.get('mean_coverage_percent', 0):.2f}% ± {cov.get('std_coverage_percent', 0):.2f}%")
        if 'ci_95_lower' in cov:
            print(f"  95% CI:            [{cov['ci_95_lower']:.2f}%, {cov['ci_95_upper']:.2f}%]")
        print(f"  Range:             {cov.get('min_coverage_percent', 0):.2f}% - {cov.get('max_coverage_percent', 0):.2f}%")
        print()

    if 'patch_metrics' in metrics:
        patch = metrics['patch_metrics']
        print("PATCH ANALYSIS:")
        print(f"  Number of Patches:   {patch.get('num_patches', 0)}")
        print(f"  Total Area:          {patch.get('total_greenspace_area_km2', 0):.4f} km²")
        print(f"  Mean Patch Size:     {patch.get('mean_patch_size_m2', 0):.2f} m²")
        print(f"  Patch Density:       {patch.get('patch_density_per_km2', 0):.2f} patches/km²")
        print()

    if 'landscape_metrics' in metrics:
        land = metrics['landscape_metrics']
        print("LANDSCAPE METRICS:")
        print(f"  Shape Index:         {land.get('landscape_shape_index', 0):.3f}")
        print(f"  Aggregation Index:   {land.get('aggregation_index', 0):.2f}%")
        print(f"  Fragmentation Index: {land.get('fragmentation_index', 0):.4f}")
        print()

    if 'accuracy_metrics' in metrics:
        acc = metrics['accuracy_metrics']
        print("ACCURACY ASSESSMENT (vs Pseudo-labels):")
        print(f"  Overall Accuracy:    {acc.get('overall_accuracy', 0):.2f}%")
        print(f"  F1 Score:            {acc.get('f1_score', 0):.4f}")
        print(f"  Cohen's Kappa:       {acc.get('cohens_kappa', 0):.4f}")
        print(f"  IoU (Jaccard):       {acc.get('iou_jaccard', 0):.4f}")
        print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - {csv_path}")
    print(f"  - {report_path}")
    print(f"  - {model_path}")
    print()
    print("Note: Accuracy metrics use pseudo-labels as reference.")
    print("For publication, replace with manually annotated ground truth.")
    print()


if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Run main analysis
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
