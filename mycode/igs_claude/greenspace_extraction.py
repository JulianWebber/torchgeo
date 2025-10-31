"""
Greenspace Extraction from Satellite Imagery
Uses TorchGeo, U-Net architecture, and GSI Seamless Photo tiles from Japan
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import requests
from PIL import Image
import io
from typing import Tuple, Dict, List, Optional
import torchvision.transforms as transforms
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets.utils import BoundingBox
import geopandas as gpd
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config


class GSISeamlessPhotoDataset(RasterDataset):
    """
    Custom TorchGeo dataset for GSI Seamless Photo tiles
    """
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (west, south, east, north) in WGS84
        zoom_level: int = None,
        cache_dir: str = "./gsi_cache",
        transforms: Optional = None
    ):
        """
        Initialize GSI Seamless Photo dataset
        
        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            zoom_level: Zoom level for tiles (18 recommended for ~10-20cm resolution)
            cache_dir: Directory to cache downloaded tiles
            transforms: Optional transforms to apply
        """
        self.bbox = bbox
        self.zoom_level = zoom_level if zoom_level is not None else config.GSI_CONFIG['default_zoom']
        self.cache_dir = cache_dir
        # Store JPGs in separate directory (NOT subdirectory of cache_dir, to avoid TorchGeo scanning them)
        self.tiles_dir = "./gsi_tiles"
        self.transforms = transforms
        self.base_url = config.GSI_CONFIG['base_url']

        # Create cache directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.tiles_dir, exist_ok=True)

        # Move any existing JPG tiles to tiles_dir (for backwards compatibility)
        import glob
        for old_tile in glob.glob(os.path.join(cache_dir, "tile_*.jpg")):
            tile_name = os.path.basename(old_tile)
            new_path = os.path.join(self.tiles_dir, tile_name)
            if not os.path.exists(new_path):
                os.rename(old_tile, new_path)
        # Also check tiles subdirectory
        tiles_subdir = os.path.join(cache_dir, "tiles")
        if os.path.exists(tiles_subdir):
            for old_tile in glob.glob(os.path.join(tiles_subdir, "tile_*.jpg")):
                tile_name = os.path.basename(old_tile)
                new_path = os.path.join(self.tiles_dir, tile_name)
                if not os.path.exists(new_path):
                    os.rename(old_tile, new_path)

        # Download tiles for the bbox BEFORE initializing parent class
        # (TorchGeo's RasterDataset expects files to exist in cache_dir)
        self._download_tiles()

        # Initialize parent class (only looks at GeoTIFFs in cache_dir, not JPGs in tiles_dir)
        super().__init__(cache_dir, crs=CRS.from_epsg(3857), res=self._calculate_resolution())
        
    def _calculate_resolution(self) -> float:
        """Calculate resolution based on zoom level"""
        # At zoom 18, resolution is approximately 0.6m
        return 156543.03392 / (2 ** self.zoom_level)
    
    def _latlon_to_tile(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to tile coordinates"""
        n = 2.0 ** self.zoom_level
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - np.log(np.tan(np.radians(lat)) + 
                              1.0 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n)
        return x, y
    
    def _tile_to_latlon(self, x: int, y: int) -> Tuple[float, float, float, float]:
        """Get bounding box of a tile in lat/lon"""
        n = 2.0 ** self.zoom_level
        lon_west = x / n * 360.0 - 180.0
        lon_east = (x + 1) / n * 360.0 - 180.0
        lat_north = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
        lat_south = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
        return lon_west, lat_south, lon_east, lat_north
    
    def _download_tile(self, x: int, y: int) -> Optional[np.ndarray]:
        """Download a single tile"""
        url = f"{self.base_url}/{self.zoom_level}/{x}/{y}.jpg"
        tile_filename = f"tile_{self.zoom_level}_{x}_{y}.jpg"
        tile_path = os.path.join(self.tiles_dir, tile_filename)

        # Check new location first, then old location for backwards compatibility
        old_tile_path = os.path.join(self.cache_dir, tile_filename)

        if os.path.exists(tile_path):
            return np.array(Image.open(tile_path))
        elif os.path.exists(old_tile_path):
            # Found in old location, use it
            return np.array(Image.open(old_tile_path))

        try:
            response = requests.get(url, timeout=config.GSI_CONFIG['timeout'])
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img.save(tile_path)
                return np.array(img)
        except Exception as e:
            print(f"Failed to download tile {x},{y}: {e}")

        return None
    
    def _download_tiles(self):
        """Download all tiles covering the bounding box"""
        west, south, east, north = self.bbox
        
        # Convert bbox corners to tile coordinates
        x_min, y_max = self._latlon_to_tile(south, west)
        x_max, y_min = self._latlon_to_tile(north, east)
        
        print(f"Downloading tiles from ({x_min}, {y_min}) to ({x_max}, {y_max})")
        
        # Create a merged image
        tiles = []
        for y in range(y_min, y_max + 1):
            row = []
            for x in range(x_min, x_max + 1):
                tile = self._download_tile(x, y)
                if tile is not None:
                    row.append(tile)
                else:
                    # Create blank tile if download fails (GSI tiles are always 256x256)
                    tile_size = config.GSI_CONFIG['tile_size']
                    row.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
            if row:
                tiles.append(np.hstack(row))
        
        if tiles:
            merged_image = np.vstack(tiles)
            
            # Calculate bounds for the merged image
            west_bound, south_bound, _, _ = self._tile_to_latlon(x_min, y_max)
            _, _, east_bound, north_bound = self._tile_to_latlon(x_max, y_min)
            
            # Save as GeoTIFF
            output_path = os.path.join(self.cache_dir, "merged.tif")
            self._save_as_geotiff(merged_image, output_path, 
                                 (west_bound, south_bound, east_bound, north_bound))
    
    def _save_as_geotiff(self, image: np.ndarray, output_path: str, bounds: Tuple[float, float, float, float]):
        """Save image as GeoTIFF with proper georeferencing"""
        west, south, east, north = bounds
        height, width = image.shape[:2]
        
        # Convert to Web Mercator (EPSG:3857)
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        west_m, south_m = transformer.transform(west, south)
        east_m, north_m = transformer.transform(east, north)
        
        transform = from_bounds(west_m, south_m, east_m, north_m, width, height)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=image.dtype,
            crs='EPSG:3857',
            transform=transform
        ) as dst:
            for i in range(3):
                dst.write(image[:, :, i], i + 1)


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 2, features: List[int] = None):
        super(UNet, self).__init__()

        if features is None:
            features = config.MODEL_CONFIG['features']
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)


class SimpleTrainingDataset(Dataset):
    """Simple dataset for training from a single merged GeoTIFF"""

    def __init__(self, geotiff_path: str, patch_size: int = None):
        self.patch_size = patch_size if patch_size is not None else config.GSI_CONFIG['tile_size']

        # Load the full image
        with rasterio.open(geotiff_path) as src:
            self.image = src.read()  # (bands, height, width)

        # Convert to HWC format
        self.image = np.transpose(self.image, (1, 2, 0))  # (height, width, channels)

        # Create pseudo-labels using vegetation indices
        self.mask = self._create_pseudo_labels(self.image)

        # Calculate number of patches
        self.height, self.width = self.image.shape[:2]
        self.patches_per_row = self.width // self.patch_size
        self.patches_per_col = self.height // self.patch_size
        self.num_patches = self.patches_per_row * self.patches_per_col

        print(f"Created training dataset with {self.num_patches} patches")

    def _create_pseudo_labels(self, image: np.ndarray) -> np.ndarray:
        """Create pseudo-labels using vegetation indices"""
        img_float = image.astype(np.float32) / 255.0

        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # Calculate vegetation indices
        exg = 2 * g - r - b
        ngrdi = (g - r) / (g + r + 1e-8)
        vari = (g - r) / (g + r - b + 1e-8)

        vegetation_score = (exg + ngrdi + vari) / 3

        # Threshold using config value
        threshold = np.percentile(vegetation_score, config.VEGETATION_INDICES['threshold_percentile'])
        greenspace_mask = (vegetation_score > threshold).astype(np.uint8)

        # Clean up with morphology
        from scipy import ndimage
        greenspace_mask = ndimage.binary_opening(greenspace_mask, iterations=1)
        greenspace_mask = ndimage.binary_closing(greenspace_mask, iterations=1)

        return greenspace_mask.astype(np.uint8)

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Calculate patch position
        row = idx // self.patches_per_row
        col = idx % self.patches_per_row

        y = row * self.patch_size
        x = col * self.patch_size

        # Extract patch
        image_patch = self.image[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = self.mask[y:y+self.patch_size, x:x+self.patch_size]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_patch).long()

        # Normalize image
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)

        return {'image': image_tensor, 'mask': mask_tensor}


class GreenspaceExtractor:
    """
    Main class for extracting greenspace from satellite imagery
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None
    ):
        # Use config for device if not specified
        if device is None:
            if config.HARDWARE_CONFIG['device'] == 'auto':
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = config.HARDWARE_CONFIG['device']

        self.device = device
        self.model = UNet(
            in_channels=config.MODEL_CONFIG['input_channels'],
            out_channels=config.MODEL_CONFIG['output_channels']
        ).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        else:
            print("Initialized new model")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_training_data(self, image: np.ndarray) -> np.ndarray:
        """
        Create pseudo-labels for training using vegetation indices
        This is a simple approach - in practice, you'd want manually labeled data
        """
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Extract color channels
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]
        
        # Calculate vegetation indices
        # Excess Green Index
        exg = 2 * g - r - b
        
        # Normalized Green-Red Difference Index
        ngrdi = (g - r) / (g + r + 1e-8)
        
        # Visible Atmospherically Resistant Index
        vari = (g - r) / (g + r - b + 1e-8)
        
        # Combine indices
        vegetation_score = (exg + ngrdi + vari) / 3
        
        # Threshold to create binary mask
        threshold = np.percentile(vegetation_score, 70)
        greenspace_mask = (vegetation_score > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up
        from scipy import ndimage
        greenspace_mask = ndimage.binary_opening(greenspace_mask, iterations=2)
        greenspace_mask = ndimage.binary_closing(greenspace_mask, iterations=2)
        
        return greenspace_mask.astype(np.uint8)
    
    def train(
        self,
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        save_path: str = "greenspace_model.pth"
    ):
        """Train the U-Net model"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device).long()
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """Predict greenspace mask for an image"""
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            output = self.model(image)
            
            # Get class predictions
            _, predicted = torch.max(output, 1)
            
            return predicted.cpu().numpy().squeeze()
    
    def extract_greenspace(
        self,
        bbox: Tuple[float, float, float, float],
        output_path: str = "greenspace_mask.tif"
    ):
        """
        Extract greenspace from satellite imagery for a given bounding box
        
        Args:
            bbox: Bounding box (west, south, east, north) in WGS84
            output_path: Path to save the greenspace mask
        """
        # Create dataset
        print("Creating GSI dataset...")
        dataset = GSISeamlessPhotoDataset(bbox)  # Uses config zoom level
        
        # Load the merged GeoTIFF and process it directly
        import rasterio
        merged_path = os.path.join(dataset.cache_dir, "merged.tif")

        with rasterio.open(merged_path) as src:
            image = src.read()  # Shape: (bands, height, width)
            transform = src.transform
            crs = src.crs

            # Convert to HWC format and normalize
            image_hwc = np.transpose(image, (1, 2, 0))  # (height, width, channels)

            # Process in patches to avoid memory issues
            print(f"Processing image of size {image_hwc.shape}...")
            height, width = image_hwc.shape[:2]
            patch_size = config.GSI_CONFIG['tile_size']

            # Create output mask
            full_mask = np.zeros((height, width), dtype=np.uint8)

            num_patches = 0
            for y in range(0, height, patch_size):
                for x in range(0, width, patch_size):
                    y_end = min(y + patch_size, height)
                    x_end = min(x + patch_size, width)

                    patch = image_hwc[y:y_end, x:x_end]

                    # Pad if needed
                    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                        padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                        padded[:patch.shape[0], :patch.shape[1]] = patch
                        patch = padded

                    # Predict
                    patch_tensor = self.transform(patch)
                    mask_patch = self.predict(patch_tensor.unsqueeze(0))

                    # Place in full mask
                    full_mask[y:y_end, x:x_end] = mask_patch[:y_end-y, :x_end-x]
                    num_patches += 1

            print(f"Processed {num_patches} patches")

        # Save results as GeoTIFF
        self._save_geotiff(full_mask, transform, crs, output_path, merged_path)
    
    def _save_geotiff(self, mask, transform, crs, output_path, original_image_path):
        """Save greenspace mask as GeoTIFF with proper georeferencing"""
        import rasterio
        from rasterio.transform import from_bounds

        print(f"Saving results to {output_path}")

        # Get dimensions
        height, width = mask.shape

        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(mask, 1)

        print(f"GeoTIFF saved successfully")

        # Also save visualization
        self._visualize_results(original_image_path, output_path)

    def _visualize_results(self, original_path, mask_path):
        """Create visualization comparing original image and greenspace mask"""
        import rasterio
        import matplotlib.pyplot as plt

        with rasterio.open(original_path) as src_orig:
            original = src_orig.read()
            # Convert to RGB for display (CHW -> HWC)
            original_rgb = np.transpose(original[:3], (1, 2, 0))

        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Satellite Image")
        axes[0].axis('off')

        # Greenspace mask
        axes[1].imshow(mask, cmap='Greens')
        axes[1].set_title("Greenspace Mask")
        axes[1].axis('off')

        # Overlay
        overlay = original_rgb.copy()
        green_overlay = np.zeros_like(overlay)
        green_overlay[:, :, 1] = mask * 255  # Green channel
        overlay = np.where(mask[:, :, None] > 0,
                          (overlay * 0.6 + green_overlay * 0.4).astype(np.uint8),
                          overlay)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        viz_path = os.path.join(config.OUTPUT_CONFIG.get('output_directory', './output'), "greenspace_extraction_results.png")
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        plt.savefig(viz_path, dpi=config.OUTPUT_CONFIG.get('visualization_dpi', 150), bbox_inches='tight')
        print(f"Results saved to {viz_path}")
        plt.close()


def main():
    """
    Main function demonstrating the greenspace extraction pipeline
    """
    # Define area of interest (Yokohama, Japan)
    # Use predefined area from config or specify custom bbox
    yokohama_bbox = config.AREAS['yokohama_station']  # (west, south, east, north)
    
    # Initialize extractor
    extractor = GreenspaceExtractor()

    # Step 1: Download tiles and create merged GeoTIFF
    print("Downloading tiles...")
    gsi_dataset = GSISeamlessPhotoDataset(yokohama_bbox)  # Uses config zoom level
    merged_path = os.path.join(gsi_dataset.cache_dir, "merged.tif")

    # Step 2: Create training dataset from merged GeoTIFF
    print("\nCreating training dataset with pseudo-labels...")
    training_dataset = SimpleTrainingDataset(merged_path)  # Uses config patch size

    # Step 3: Train the model
    print("\nTraining U-Net model...")
    model_save_path = os.path.join(config.OUTPUT_CONFIG.get('output_directory', './output'), "greenspace_unet.pth")
    os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else '.', exist_ok=True)

    extractor.train(
        training_dataset,
        epochs=config.TRAINING_CONFIG['epochs'],
        batch_size=config.TRAINING_CONFIG['batch_size'],
        learning_rate=config.TRAINING_CONFIG['learning_rate'],
        save_path=model_save_path
    )

    # Step 4: Extract greenspace using trained model
    print("\nExtracting greenspace...")
    output_path = os.path.join(config.OUTPUT_CONFIG.get('output_directory', './output'), "yokohama_greenspace.tif")
    extractor.extract_greenspace(yokohama_bbox, output_path)

    viz_path = os.path.join(config.OUTPUT_CONFIG.get('output_directory', './output'), "greenspace_extraction_results.png")
    print(f"\nDone! Check {output_path} and {viz_path}")


def visualize_results(bbox: Tuple[float, float, float, float]):
    """Visualize the extracted greenspace"""
    # Load the original image and mask
    # This is a placeholder - implement based on your saved results
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axes[0].set_title("Original Satellite Image")
    axes[0].axis('off')
    
    # Plot greenspace mask
    axes[1].set_title("Extracted Greenspace")
    axes[1].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(config.OUTPUT_CONFIG.get('output_directory', './output'), "greenspace_extraction_results.png")
    os.makedirs(os.path.dirname(viz_path), exist_ok=True)
    plt.savefig(viz_path, dpi=config.OUTPUT_CONFIG.get('visualization_dpi', 150), bbox_inches='tight')
    print(f"Results saved to {viz_path}")


if __name__ == "__main__":
    # Install required packages
    print("Installing required packages...")
    os.system("pip install torchgeo torch torchvision rasterio geopandas requests pillow matplotlib scipy pyproj --break-system-packages")
    
    # Run main pipeline
    main()
