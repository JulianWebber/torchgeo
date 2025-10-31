# Greenspace Statistics Guide for Journal Publications

This guide explains all the statistics calculated by the greenspace extraction system and their relevance for remote sensing journal publications.

## Overview

The statistics module (`greenspace_statistics.py`) calculates comprehensive metrics across four categories:

1. **Classification Accuracy Metrics** - Model performance assessment
2. **Coverage Statistics** - Descriptive statistics of greenspace distribution
3. **Patch Metrics** - Spatial configuration analysis
4. **Landscape Metrics** - Fragmentation and connectivity indices

## Usage

### Basic Usage

```python
from greenspace_statistics import GreenspaceStatistics, export_metrics_to_csv

# Initialize calculator (10m resolution for zoom level 18)
stats = GreenspaceStatistics(pixel_resolution_m=10.0)

# Calculate all metrics
metrics = stats.calculate_all_metrics(
    predictions=prediction_masks,
    ground_truth=ground_truth_masks,  # Optional, for accuracy assessment
    bbox=(west, south, east, north),
    metadata={'model': 'U-Net', 'area': 'Yokohama'}
)

# Export to CSV
export_metrics_to_csv(metrics, 'output/greenspace_statistics.csv')
```

### Quick Start

```bash
# Run with synthetic demo data
python demo_statistics.py

# Run with real data (requires training)
python run_with_statistics.py
```

## Metrics Explanation

### 1. Classification Accuracy Metrics

These metrics assess model performance when ground truth is available.

#### Overall Accuracy (OA)
- **Formula**: (TP + TN) / Total
- **Range**: 0-100%
- **Interpretation**: Percentage of correctly classified pixels
- **Journal Standard**: Report this as primary accuracy metric

#### Producer's Accuracy (Recall/Sensitivity)
- **Formula**: TP / (TP + FN)
- **Range**: 0-100%
- **Interpretation**: Percentage of actual greenspace correctly identified
- **Journal Relevance**: Shows omission errors (missed greenspace)

#### User's Accuracy (Precision)
- **Formula**: TP / (TP + FP)
- **Range**: 0-100%
- **Interpretation**: Percentage of predicted greenspace that is correct
- **Journal Relevance**: Shows commission errors (false greenspace)

#### F1 Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0-1
- **Interpretation**: Harmonic mean of precision and recall
- **Journal Standard**: Commonly reported metric, especially for imbalanced datasets

#### Cohen's Kappa (κ)
- **Formula**: (Po - Pe) / (1 - Pe)
- **Range**: -1 to 1 (typically 0-1)
- **Interpretation**: Agreement beyond chance
  - < 0.0: Poor agreement
  - 0.0-0.2: Slight agreement
  - 0.2-0.4: Fair agreement
  - 0.4-0.6: Moderate agreement
  - 0.6-0.8: Substantial agreement
  - 0.8-1.0: Almost perfect agreement
- **Journal Standard**: Required for most remote sensing publications

#### IoU (Intersection over Union) / Jaccard Index
- **Formula**: TP / (TP + FP + FN)
- **Range**: 0-1
- **Interpretation**: Overlap between prediction and ground truth
- **Journal Relevance**: Standard metric for segmentation tasks

#### Specificity
- **Formula**: TN / (TN + FP)
- **Range**: 0-100%
- **Interpretation**: Ability to correctly identify non-greenspace

### 2. Coverage Statistics

Basic descriptive statistics of greenspace distribution across tiles.

#### Mean Coverage
- Average percentage of greenspace across all tiles
- **Journal Use**: Primary measure of greenspace abundance

#### Median Coverage
- Middle value of coverage distribution
- **Journal Use**: Robust measure less affected by outliers

#### Standard Deviation
- Spread of coverage values
- **Journal Use**: Indicates heterogeneity of greenspace

#### Coefficient of Variation (CV)
- **Formula**: σ / μ
- Normalized measure of variability
- **Journal Use**: Compare variability across different areas

#### 95% Confidence Interval
- Range likely to contain true mean
- **Journal Standard**: Always report confidence intervals

#### Quartiles (Q1, Q3, IQR)
- 25th percentile, 75th percentile, and interquartile range
- **Journal Use**: Describe distribution shape

### 3. Patch Metrics

Analyzes spatial configuration of greenspace patches.

#### Number of Patches
- Count of distinct connected greenspace regions
- **Interpretation**: Higher values indicate more fragmentation
- **Journal Use**: Primary fragmentation indicator

#### Total Greenspace Area (km²)
- Sum of all greenspace in the study area
- **Journal Standard**: Always report absolute area

#### Mean/Median Patch Size (m²)
- Average/middle size of greenspace patches
- **Interpretation**:
  - Large patches: Contiguous, well-connected greenspace
  - Small patches: Fragmented landscape
- **Journal Use**: Core landscape metric

#### Largest Patch Size (m²)
- Size of the biggest greenspace patch
- **Interpretation**: Indicates presence of large green infrastructure
- **Journal Use**: Important for habitat connectivity studies

#### Patch Density (patches/km²)
- **Formula**: Number of patches / Total area
- **Interpretation**:
  - Low density: Few large patches
  - High density: Many small patches (fragmentation)
- **Journal Standard**: Standard fragmentation metric

#### Edge Density (m/ha)
- **Formula**: Total edge length / Area
- **Interpretation**: Amount of greenspace edge relative to area
  - Low values: Compact patches
  - High values: Complex, fragmented patches
- **Journal Use**: Landscape ecology studies

#### Mean Patch Perimeter (m)
- Average perimeter length of patches
- **Journal Use**: Shape complexity indicator

### 4. Landscape Metrics

Advanced indices for landscape-level analysis.

#### Landscape Shape Index (LSI)
- **Formula**: Total edge / Minimum edge
- **Range**: ≥1 (1 = perfect circle)
- **Interpretation**:
  - LSI ≈ 1: Simple, compact shapes
  - LSI > 2: Complex, irregular shapes
  - Higher values: More fragmentation and edge complexity
- **Journal Use**: Shape complexity and fragmentation

#### Aggregation Index (AI)
- **Formula**: (Largest patch area / Total area) × 100
- **Range**: 0-100%
- **Interpretation**:
  - High values (>80%): Highly aggregated/clustered
  - Low values (<20%): Dispersed/fragmented
- **Journal Use**: Measures clustering of greenspace

#### Fragmentation Index
- **Formula**: 1 / (Mean patch size / 1000)
- **Interpretation**: Inverse of mean patch size
  - Higher values: More fragmented
  - Lower values: More contiguous
- **Journal Use**: Simple fragmentation measure

#### Largest Patch Index (LPI)
- **Formula**: (Largest patch / Total landscape) × 100
- **Range**: 0-100%
- **Interpretation**:
  - High LPI: Dominated by single large patch
  - Low LPI: No dominant patch (fragmentation)
- **Journal Standard**: Core landscape ecology metric

#### Effective Mesh Size (MESH)
- **Formula**: Σ(patch_area²) / Total_area
- **Units**: km²
- **Interpretation**: Area-weighted mean patch size
  - Large values: Connected landscape
  - Small values: Fragmented landscape
- **Journal Use**: Habitat connectivity research

## Output Files

### CSV File (`greenspace_statistics.csv`)

- One row per analysis
- All metrics in columns with descriptive headers
- Format: `category_metric_name` (e.g., `patch_metrics_num_patches`)
- **Use**: Import into statistical software (R, Python, Excel) for analysis
- **Journal Submission**: Include as supplementary material

### Text Report (`greenspace_report.txt`)

- Human-readable formatted report
- Organized by metric category
- Includes metadata and timestamp
- **Use**: Quick reference and report sharing

## Journal Publication Guidelines

### Required Metrics (Minimum)

For a standard remote sensing journal paper, include:

1. **Accuracy Assessment**:
   - Overall Accuracy
   - Cohen's Kappa
   - F1 Score
   - Confusion Matrix (as table)

2. **Coverage**:
   - Mean coverage ± std deviation
   - 95% confidence interval
   - Total greenspace area (km²)

3. **Spatial Analysis**:
   - Number of patches
   - Mean patch size
   - Patch density

### Recommended Additional Metrics

For higher-impact journals (e.g., Remote Sensing of Environment, ISPRS):

4. **Landscape Metrics**:
   - Landscape Shape Index
   - Largest Patch Index
   - Effective Mesh Size

5. **Fragmentation Analysis**:
   - Edge density
   - Aggregation Index

### Reporting Format

Example paragraph for Methods section:

> "Classification accuracy was assessed using overall accuracy (OA), Cohen's kappa (κ), and F1 score. Spatial configuration of greenspace was analyzed using patch-based metrics including number of patches, mean patch size, and patch density. Landscape-level metrics including the Landscape Shape Index (LSI), Largest Patch Index (LPI), and Effective Mesh Size (MESH) were calculated to assess fragmentation. All spatial metrics were computed at 10m resolution corresponding to zoom level 18 of the GSI Seamless Photo tiles."

Example paragraph for Results section:

> "The U-Net model achieved an overall accuracy of 76.1% (κ=0.464, F1=0.830) when compared against vegetation index-based pseudo-labels. Mean greenspace coverage was 58.4% ± 48.0% (95% CI: 36.8%-79.9%). The study area contained 5 distinct greenspace patches with a mean size of 15.3 × 10⁶ m² and patch density of 0.04 patches/km². The Landscape Shape Index of 2.91 indicated moderate shape complexity, while the Aggregation Index of 48.6% suggested moderate clustering of greenspace."

### Tables

**Table 1: Classification Accuracy Metrics**

| Metric | Value |
|--------|-------|
| Overall Accuracy (%) | XX.XX |
| Producer's Accuracy (%) | XX.XX |
| User's Accuracy (%) | XX.XX |
| F1 Score | X.XXXX |
| Cohen's Kappa | X.XXXX |
| IoU (Jaccard) | X.XXXX |

**Table 2: Greenspace Configuration Metrics**

| Metric | Value | Unit |
|--------|-------|------|
| Total Area | XX.XX | km² |
| Mean Coverage | XX.XX ± XX.XX | % |
| Number of Patches | XXX | - |
| Mean Patch Size | XXXXX | m² |
| Patch Density | X.XX | patches/km² |
| Edge Density | X.XX | m/ha |
| LSI | X.XXX | - |
| LPI | XX.XX | % |
| MESH | X.XXXX | km² |

## Notes on Interpretation

### Pseudo-labels vs Ground Truth

The current implementation uses pseudo-labels (vegetation indices) as ground truth for demonstration. For journal publication:

1. **Replace pseudo-labels with manually annotated ground truth**
2. Note the accuracy metrics in demo output compare model predictions against pseudo-labels
3. This gives an estimate of model improvement over the baseline method

### Spatial Resolution

- Default: 10m/pixel (GSI zoom 18)
- Affects area calculations
- Report resolution in methods section

### Statistical Significance

For journal publication, consider:
- Comparing multiple areas (ANOVA)
- Temporal analysis (repeated measures)
- Validation against independent data

## Citation

When using these metrics in publications, cite relevant sources:

- **Overall Accuracy, Kappa**: Congalton & Green (2019). Assessing the Accuracy of Remotely Sensed Data
- **Landscape Metrics**: McGarigal et al. (2012). FRAGSTATS documentation
- **IoU**: Jaccard (1912). The distribution of the flora in the alpine zone
- **Effective Mesh Size**: Jaeger (2000). Landscape division, splitting index, and effective mesh size

## Support

For questions about metrics or interpretation:
- See `greenspace_statistics.py` docstrings
- Run `python demo_statistics.py` for examples
- Check academic literature on landscape ecology metrics
