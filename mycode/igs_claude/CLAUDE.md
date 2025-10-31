# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a greenspace extraction system that uses deep learning (U-Net architecture) to identify vegetation from high-resolution satellite imagery. The system downloads GSI (Geospatial Information Authority of Japan) Seamless Photo tiles, generates training labels using vegetation indices, trains a U-Net model, and produces greenspace segmentation masks.

## Running the Code

### Quick Start
```bash
# Run simplified version (recommended for testing)
python greenspace_simple.py

# Or use the demo runner
python run_demo.py

# For full TorchGeo integration
python run_demo.py --full
```

### Installation
```bash
# Simplified version (minimal dependencies)
pip install torch torchvision numpy matplotlib pillow requests scipy

# Full version (includes TorchGeo and GIS capabilities)
pip install torchgeo rasterio geopandas pyproj torch torchvision numpy matplotlib pillow requests scipy
```

## Architecture Overview

The system consists of four main components that work together in a pipeline:

### 1. Data Acquisition Layer (GSI Tile Download)
- Downloads satellite imagery from GSI Seamless Photo service (zoom level 18 = ~10-20cm resolution)
- Coordinate conversion: WGS84 lat/lon ↔ Web Mercator tile indices
- Tile caching to avoid redundant downloads
- Implementation: `TileManager` class in utils.py, `download_gsi_tile()` in greenspace_simple.py, `GSISeamlessPhotoDataset` class in greenspace_extraction.py

### 2. Pseudo-labeling Layer (Vegetation Indices)
- Creates training labels automatically using RGB-based vegetation indices
- Combines multiple indices (ExG, VARI, GLI, NGRDI, ExG-ExR) for robust detection
- Adaptive thresholding using percentile-based cutoffs
- Post-processing: morphological operations to clean up noise
- Implementation: `VegetationIndices` class in utils.py, `create_vegetation_mask()` in greenspace_simple.py, `create_training_data()` in greenspace_extraction.py

**Key indices used:**
- ExG (Excess Green): 2G - R - B
- VARI: (G - R) / (G + R - B)
- GLI: (2G - R - B) / (2G + R + B)
- NGRDI: (G - R) / (G + R)

### 3. Model Layer (U-Net)
- Encoder-decoder architecture with skip connections
- 4 encoder levels: 64 → 128 → 256 → 512 features
- Bottleneck: 1024 features
- Symmetric decoder with skip connections from encoder
- Binary segmentation output (greenspace/non-greenspace)
- Implementation: `SimpleUNet` class in greenspace_simple.py, `UNet` class in greenspace_extraction.py

### 4. Analysis & Visualization Layer
- Statistical analysis: greenspace coverage percentages
- Report generation with multiple visualizations
- GeoJSON export for GIS software integration
- Implementation: `DataVisualizer` class in utils.py, `visualize_results()` functions

## File Structure

- **greenspace_simple.py**: Standalone simplified version, good starting point for understanding the workflow
- **greenspace_extraction.py**: Full version with TorchGeo integration, GeoTIFF export, proper georeferencing
- **utils.py**: Reusable utilities (coordinate conversion, vegetation indices, visualization, tile management)
- **config.py**: All configuration parameters (model settings, training hyperparameters, vegetation thresholds, predefined areas)
- **run_demo.py**: Entry point script that runs either simple or full version

## Configuration

All settings are centralized in `config.py`:

- **GSI_CONFIG**: Tile server URL, zoom level, timeout settings
- **AREAS**: Pre-defined bounding boxes (Yokohama Station, Yokohama Park, Minato Mirai, Yamashita Park)
- **MODEL_CONFIG**: U-Net architecture parameters
- **TRAINING_CONFIG**: Batch size (4), epochs (10), learning rate (0.001)
- **VEGETATION_INDICES**: Index weights and threshold percentile (60 = moderate sensitivity)
- **OUTPUT_CONFIG**: Output formats and visualization settings

To adjust detection sensitivity, modify `threshold_percentile` in VEGETATION_INDICES (lower = more vegetation detected).

## Key Design Patterns

### Two-Track Implementation
The codebase provides two parallel implementations:
- **Simple track** (greenspace_simple.py): Minimal dependencies, easier to understand, uses basic NumPy arrays
- **Full track** (greenspace_extraction.py): TorchGeo integration, proper georeferencing, GeoTIFF export with CRS

This allows users to start simple and graduate to the full version when needed.

### Pseudo-labeling Strategy
Since manual annotation is expensive, the system generates training labels automatically using vegetation indices. This is a bootstrapping approach:
1. Calculate multiple vegetation indices from RGB
2. Combine weighted indices into vegetation score
3. Apply adaptive thresholding (percentile-based)
4. Clean with morphological operations
5. Use as training labels for U-Net

**Limitation**: Accuracy is ~85% with pseudo-labels. For production use, replace with manually annotated ground truth.

### Progressive Tiling
Large areas are processed tile-by-tile to manage memory:
1. Convert bounding box to tile grid
2. Download/cache tiles individually
3. Train on tile patches
4. Predict tile-by-tile
5. Mosaic results

## Data Flow

```
User Input (bbox) → Tile Coordinates → Download Tiles → Cache
                                                          ↓
                                               RGB Image Tiles
                                                          ↓
                          ┌───────────────────────────────┴──────────────┐
                          ↓                                              ↓
              Calculate Vegetation Indices                    U-Net Model Input
                          ↓                                              ↓
            Adaptive Thresholding + Morphology                  Forward Pass
                          ↓                                              ↓
                Pseudo-label Masks                         Predicted Masks
                          ↓                                              ↓
                          └──────────────→ Training ←────────────────────┘
                                              ↓
                                         Trained Model
                                              ↓
                                      Greenspace Analysis
                                              ↓
                          ┌───────────────────┴───────────────────┐
                          ↓                                       ↓
                  Visualization Reports                  GeoJSON/GeoTIFF Export
```

## Important Constraints

1. **RGB-only**: GSI tiles are RGB. For better accuracy, NIR (near-infrared) bands would help but aren't available.

2. **Pseudo-labels**: Color-based indices have limitations (shadows, water reflections, seasonal variation). Manual labels would significantly improve accuracy.

3. **No cloud detection**: Assumes cloud-free imagery. Cloudy tiles will produce incorrect results.

4. **Zoom level 18**: Hardcoded to ~10-20cm resolution. Lower zoom levels process faster but sacrifice detail.

5. **Japan-specific**: GSI Seamless Photo only covers Japan. For other regions, replace tile source in GSI_CONFIG.

## Common Modifications

### Change Target Area
Edit bounding box in the script or add to AREAS in config.py:
```python
# Format: (west, south, east, north) in WGS84
custom_bbox = (139.615, 35.461, 139.628, 35.471)
```

### Adjust Sensitivity
In config.py:
```python
VEGETATION_INDICES = {
    'threshold_percentile': 60,  # Lower = more vegetation detected (40-80 typical range)
}
```

### Change Model Architecture
In config.py:
```python
MODEL_CONFIG = {
    'features': [64, 128, 256, 512],  # Increase for more capacity
}
```

### Add New Vegetation Index
In utils.py VegetationIndices class, add new static method, then update combined_index() to include it.

## Output Files

- **greenspace_predictions.png**: Visual comparison of original images, pseudo-labels, and U-Net predictions
- **greenspace_unet.pth**: Trained model weights (can be reloaded)
- **greenspace_report.png**: Comprehensive analysis with statistics and histograms
- **greenspace.geojson**: Vector format for GIS software (QGIS, ArcGIS)
- **merged.tif** or **yokohama_greenspace.tif**: Georeferenced raster output

## Dependencies Note

The system has two dependency tiers:
- **Core**: torch, torchvision, numpy, matplotlib, pillow, requests, scipy (always needed)
- **GIS**: torchgeo, rasterio, geopandas, pyproj (only for full version with georeferencing)

Scripts automatically attempt pip install with --break-system-packages flag. On some systems this may fail; remove the flag if using virtual environments.

## Performance Characteristics

- Training time: ~30 seconds for 5 epochs on small area (CPU)
- GPU acceleration: 5-10x faster with CUDA
- Tile download: Bottleneck is network latency (~100-200ms per tile)
- Memory: ~2GB for typical small area, scales with number of tiles
- Batch size 4 is conservative; increase if GPU memory allows

## Testing/Development Workflow

1. Start with small bounding box (0.01° × 0.01° = ~1km²) to iterate quickly
2. Use greenspace_simple.py for rapid prototyping
3. Adjust threshold_percentile until pseudo-labels look reasonable
4. Train for 5 epochs initially, increase if loss hasn't plateaued
5. Visually inspect predictions in greenspace_predictions.png
6. Scale up to larger area once satisfied
7. Switch to greenspace_extraction.py for production with GIS export
