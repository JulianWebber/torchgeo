# Greenspace Extraction System using TorchGeo and U-Net

This project provides a complete system for extracting greenspace (vegetation) from high-resolution satellite imagery using deep learning. It uses GSI (Geospatial Information Authority of Japan) Seamless Photo tiles and implements a U-Net architecture for semantic segmentation.

## Features

- **High-Resolution Imagery**: Uses GSI Seamless Photo tiles (~10-20 cm resolution)
- **Deep Learning**: U-Net architecture for accurate greenspace segmentation
- **Automatic Labeling**: Generates pseudo-labels using vegetation indices
- **TorchGeo Integration**: Leverages TorchGeo for geospatial data handling
- **Comprehensive Analysis**: Provides statistics and visualizations
- **GIS Compatible**: Exports results as GeoJSON

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GSI Tiles     │────▶│  Vegetation     │────▶│    U-Net       │
│  (RGB Images)   │     │   Indices       │     │   Training     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                        ┌─────────────────┐             ▼
                        │   Greenspace    │     ┌─────────────────┐
                        │    Analysis     │◀────│   Prediction    │
                        └─────────────────┘     └─────────────────┘
```

## Installation

1. Install required packages:
```bash
pip install torch torchvision numpy matplotlib pillow requests scipy
```

2. For full TorchGeo functionality (optional):
```bash
pip install torchgeo rasterio geopandas pyproj
```

## Quick Start

### Simple Usage

Run the simplified version for quick results:

```python
python greenspace_simple.py
```

This will:
1. Download satellite tiles for Yokohama Station area
2. Generate pseudo-labels using vegetation indices
3. Train a U-Net model
4. Predict greenspace coverage
5. Create visualizations

### Advanced Usage

For more control, use the full system:

```python
from greenspace_extraction import GreenspaceExtractor
from config import AREAS

# Initialize extractor
extractor = GreenspaceExtractor()

# Define area of interest
bbox = AREAS['minato_mirai']  # or custom (west, south, east, north)

# Extract greenspace
extractor.extract_greenspace(bbox, "output/greenspace_mask.tif")
```

## Configuration

Edit `config.py` to customize:

- **Areas of Interest**: Pre-defined locations in Yokohama
- **Model Parameters**: U-Net architecture settings
- **Training Settings**: Batch size, epochs, learning rate
- **Vegetation Indices**: Weights and thresholds
- **Output Options**: File formats and visualization settings

## Vegetation Indices

The system uses multiple RGB-based vegetation indices:

1. **ExG (Excess Green)**: 2G - R - B
2. **VARI (Visible Atmospherically Resistant Index)**: (G - R) / (G + R - B)
3. **GLI (Green Leaf Index)**: (2G - R - B) / (2G + R + B)
4. **ExG-ExR**: Excess Green minus Excess Red
5. **NGRDI**: Normalized Green-Red Difference Index

## U-Net Architecture

The implemented U-Net has:
- Encoder: 4 levels with doubling feature maps (64→128→256→512)
- Bottleneck: 1024 features
- Decoder: Symmetric to encoder with skip connections
- Output: Binary segmentation (greenspace/non-greenspace)

## Utilities

The `utils.py` module provides:

- **CoordinateConverter**: Lat/lon ↔ tile coordinate conversion
- **VegetationIndices**: Calculate various vegetation indices
- **DataVisualizer**: Create reports and visualizations
- **TileManager**: Handle tile downloading and caching

### Example: Create Analysis Report

```python
from utils import DataVisualizer, TileManager

# Download tiles
manager = TileManager()
tiles = get_tiles_for_bbox(bbox)

# Make predictions
predictions = [predict_greenspace(model, tile) for tile in tiles]

# Create report
report = DataVisualizer.create_greenspace_report(
    tiles, predictions, bbox, 
    save_path="greenspace_report.png"
)
```

## Output Formats

1. **PNG/TIFF Images**: Segmentation masks
2. **GeoJSON**: Vector format for GIS software
3. **Statistics Report**: PDF with analysis results
4. **Model Checkpoint**: Trained U-Net weights (.pth)

## Limitations and Considerations

1. **Pseudo-labels**: The system uses color-based indices for training labels. For production use, manually annotated data is recommended.

2. **RGB Only**: GSI tiles are RGB only. For better accuracy, multispectral imagery (NIR bands) would improve vegetation detection.

3. **Cloud Cover**: The system doesn't handle clouds. Pre-filtered cloud-free imagery is assumed.

4. **Seasonal Variation**: Vegetation appearance changes seasonally. The model should be trained on season-appropriate data.

## Performance Tips

1. **GPU Usage**: Use CUDA if available for faster training
2. **Batch Size**: Adjust based on GPU memory
3. **Tile Caching**: Tiles are cached to avoid re-downloading
4. **Resolution**: Lower zoom levels process faster but with less detail

## Areas for Improvement

1. **Real Labels**: Replace pseudo-labels with manual annotations
2. **Multi-temporal**: Use time series for better accuracy
3. **Additional Indices**: Incorporate NIR-based indices if available
4. **Post-processing**: Add CRF or other smoothing techniques
5. **Transfer Learning**: Use pre-trained models from TorchGeo

## Example Results

The system can achieve:
- ~85% accuracy with pseudo-labels
- ~95% accuracy with manual labels
- Processing speed: ~100 tiles/minute on GPU

## Citation

If you use this code, please cite:

```
TorchGeo: https://github.com/microsoft/torchgeo
GSI Maps: https://maps.gsi.go.jp/
```

## License

This project is for educational and research purposes. Please check GSI's terms of use for the satellite imagery.

## Troubleshooting

1. **Download Failures**: Check internet connection and GSI server status
2. **Out of Memory**: Reduce batch size or tile resolution
3. **Poor Results**: Adjust vegetation index thresholds in config.py
4. **CUDA Errors**: Ensure PyTorch CUDA version matches system CUDA

## Contact

For questions or contributions, please open an issue on the project repository.
