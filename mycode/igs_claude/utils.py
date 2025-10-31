"""
Utility functions for Greenspace Extraction System
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import json
import os
from datetime import datetime
import requests
from PIL import Image
import io


class CoordinateConverter:
    """Utility class for coordinate conversions"""
    
    @staticmethod
    def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates"""
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - np.log(np.tan(np.radians(lat)) + 
                            1.0 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n)
        return x, y
    
    @staticmethod
    def tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
        """Get bounding box of a tile in lat/lon"""
        n = 2.0 ** zoom
        lon_west = x / n * 360.0 - 180.0
        lon_east = (x + 1) / n * 360.0 - 180.0
        lat_north = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
        lat_south = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 1) / n))))
        return lon_west, lat_south, lon_east, lat_north
    
    @staticmethod
    def calculate_area_km2(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate approximate area of a bounding box in km²"""
        west, south, east, north = bbox
        
        # Approximate using average latitude
        avg_lat = (south + north) / 2
        
        # Calculate distances
        lat_dist = (north - south) * 111.32  # km per degree latitude
        lon_dist = (east - west) * 111.32 * np.cos(np.radians(avg_lat))
        
        return lat_dist * lon_dist


class VegetationIndices:
    """Calculate various vegetation indices from RGB imagery"""
    
    @staticmethod
    def excess_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Excess Green Index (ExG)"""
        return 2 * g - r - b
    
    @staticmethod
    def excess_green_red(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Excess Green minus Excess Red (ExG-ExR)"""
        exg = 2 * g - r - b
        exr = 1.4 * r - g
        return exg - exr
    
    @staticmethod
    def vari(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Visible Atmospherically Resistant Index (VARI)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            vari = (g - r) / (g + r - b + 1e-8)
            return np.nan_to_num(vari, 0)
    
    @staticmethod
    def gli(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Green Leaf Index (GLI)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            gli = (2 * g - r - b) / (2 * g + r + b + 1e-8)
            return np.nan_to_num(gli, 0)
    
    @staticmethod
    def ngrdi(r: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Normalized Green-Red Difference Index (NGRDI)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            ngrdi = (g - r) / (g + r + 1e-8)
            return np.nan_to_num(ngrdi, 0)
    
    @staticmethod
    def combined_index(image: np.ndarray, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Combine multiple vegetation indices"""
        if weights is None:
            weights = {
                'exg': 1.0,
                'vari': 1.0,
                'gli': 1.0
            }
        
        # Normalize image
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # Calculate indices
        indices = {
            'exg': VegetationIndices.excess_green(r, g, b),
            'vari': VegetationIndices.vari(r, g, b),
            'gli': VegetationIndices.gli(r, g, b),
            'exgr': VegetationIndices.excess_green_red(r, g, b),
            'ngrdi': VegetationIndices.ngrdi(r, g)
        }
        
        # Weighted combination
        combined = np.zeros_like(r)
        total_weight = 0
        
        for idx_name, weight in weights.items():
            if idx_name in indices:
                combined += weight * indices[idx_name]
                total_weight += weight
        
        if total_weight > 0:
            combined /= total_weight
        
        return combined


class DataVisualizer:
    """Visualization utilities for greenspace analysis"""
    
    @staticmethod
    def plot_vegetation_indices(image: np.ndarray, save_path: Optional[str] = None):
        """Plot various vegetation indices for an image"""
        img = image.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # Calculate indices
        indices = {
            'Original': image,
            'ExG': VegetationIndices.excess_green(r, g, b),
            'VARI': VegetationIndices.vari(r, g, b),
            'GLI': VegetationIndices.gli(r, g, b),
            'ExG-ExR': VegetationIndices.excess_green_red(r, g, b),
            'NGRDI': VegetationIndices.ngrdi(r, g)
        }
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, data) in enumerate(indices.items()):
            ax = axes[idx]
            if name == 'Original':
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap='RdYlGn', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(name)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_greenspace_report(
        tiles: List[np.ndarray], 
        predictions: List[np.ndarray],
        bbox: Tuple[float, float, float, float],
        save_path: str = "greenspace_report.png"
    ):
        """Create a comprehensive greenspace analysis report"""
        # Calculate statistics
        greenspace_percentages = []
        for pred in predictions:
            percentage = (np.sum(pred > 0) / pred.size) * 100
            greenspace_percentages.append(percentage)
        
        avg_greenspace = np.mean(greenspace_percentages)
        total_area = CoordinateConverter.calculate_area_km2(bbox)
        greenspace_area = total_area * (avg_greenspace / 100)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # Title
        fig.suptitle(f'Greenspace Analysis Report\nArea: {bbox}', fontsize=16, y=0.98)
        
        # Grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Sample images and predictions
        for i in range(min(4, len(tiles))):
            ax_img = fig.add_subplot(gs[0, i])
            ax_img.imshow(tiles[i])
            ax_img.set_title(f'Tile {i+1}')
            ax_img.axis('off')
            
            ax_pred = fig.add_subplot(gs[1, i])
            ax_pred.imshow(predictions[i], cmap='Greens')
            ax_pred.set_title(f'{greenspace_percentages[i]:.1f}% Green')
            ax_pred.axis('off')
        
        # Statistics
        ax_stats = fig.add_subplot(gs[2, :2])
        stats_text = f"""
        Total Area: {total_area:.2f} km²
        Average Greenspace: {avg_greenspace:.1f}%
        Greenspace Area: {greenspace_area:.2f} km²
        Number of Tiles: {len(tiles)}
        Min Coverage: {np.min(greenspace_percentages):.1f}%
        Max Coverage: {np.max(greenspace_percentages):.1f}%
        Std Deviation: {np.std(greenspace_percentages):.1f}%
        """
        ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                     fontsize=12, verticalalignment='center')
        ax_stats.axis('off')
        
        # Histogram
        ax_hist = fig.add_subplot(gs[2, 2:])
        ax_hist.hist(greenspace_percentages, bins=20, edgecolor='black', alpha=0.7)
        ax_hist.set_xlabel('Greenspace Coverage (%)')
        ax_hist.set_ylabel('Number of Tiles')
        ax_hist.set_title('Distribution of Greenspace Coverage')
        ax_hist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to {save_path}")
        
        return {
            'total_area_km2': total_area,
            'avg_greenspace_percent': avg_greenspace,
            'greenspace_area_km2': greenspace_area,
            'num_tiles': len(tiles),
            'timestamp': datetime.now().isoformat()
        }


class TileManager:
    """Manage downloading and caching of GSI tiles"""
    
    def __init__(self, cache_dir: str = "./tile_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_tile_path(self, x: int, y: int, z: int) -> str:
        """Get the cache path for a tile"""
        return os.path.join(self.cache_dir, f"z{z}_x{x}_y{y}.jpg")
    
    def download_tile(self, x: int, y: int, z: int = 18) -> Optional[np.ndarray]:
        """Download a tile with caching"""
        cache_path = self.get_tile_path(x, y, z)
        
        # Check cache
        if os.path.exists(cache_path):
            return np.array(Image.open(cache_path))
        
        # Download
        url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
        
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                img.save(cache_path)
                return np.array(img)
        except Exception as e:
            print(f"Error downloading tile {x},{y},{z}: {e}")
        
        return None
    
    def clear_cache(self):
        """Clear the tile cache"""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        print("Tile cache cleared")


def save_results_geojson(
    predictions: List[np.ndarray],
    bbox: Tuple[float, float, float, float],
    output_path: str = "greenspace.geojson"
):
    """Save predictions as GeoJSON for GIS compatibility"""
    features = []
    
    # Calculate tile positions
    west, south, east, north = bbox
    x_min, y_max = CoordinateConverter.latlon_to_tile(south, west, 18)
    x_max, y_min = CoordinateConverter.latlon_to_tile(north, east, 18)
    
    tile_idx = 0
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if tile_idx < len(predictions):
                # Get tile bounds
                tile_bbox = CoordinateConverter.tile_to_latlon(x, y, 18)
                
                # Calculate greenspace percentage
                pred = predictions[tile_idx]
                greenspace_pct = (np.sum(pred > 0) / pred.size) * 100
                
                # Create GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [tile_bbox[0], tile_bbox[1]],  # SW
                            [tile_bbox[2], tile_bbox[1]],  # SE
                            [tile_bbox[2], tile_bbox[3]],  # NE
                            [tile_bbox[0], tile_bbox[3]],  # NW
                            [tile_bbox[0], tile_bbox[1]]   # Close
                        ]]
                    },
                    "properties": {
                        "tile_x": x,
                        "tile_y": y,
                        "zoom": 18,
                        "greenspace_percent": round(greenspace_pct, 2)
                    }
                }
                features.append(feature)
                tile_idx += 1
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"GeoJSON saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Greenspace Extraction Utilities")
    print("=" * 50)
    
    # Test coordinate conversion
    lat, lon = 35.465, 139.622  # Yokohama
    x, y = CoordinateConverter.latlon_to_tile(lat, lon, 18)
    print(f"Coordinates ({lat}, {lon}) -> Tile ({x}, {y}) at zoom 18")
    
    # Test area calculation
    bbox = (139.615, 35.461, 139.628, 35.471)
    area = CoordinateConverter.calculate_area_km2(bbox)
    print(f"Bounding box area: {area:.2f} km²")
