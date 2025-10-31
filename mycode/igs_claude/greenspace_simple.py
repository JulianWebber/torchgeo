"""
Simplified Greenspace Extraction using TorchGeo and U-Net
This version includes pseudo-labeling for demonstration
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import rasterio
import requests
from PIL import Image
import io
from typing import Tuple, List, Optional
import torchvision.transforms as transforms


class SimpleUNet(nn.Module):
    """Simplified U-Net for binary segmentation (greenspace vs non-greenspace)"""
    
    def __init__(self, in_channels=3, out_channels=2):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


class GSITileDataset(Dataset):
    """Dataset for GSI Seamless Photo tiles with automatic pseudo-labeling"""
    
    def __init__(self, tiles: List[np.ndarray], transform=None):
        self.tiles = tiles
        self.transform = transform
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        image = self.tiles[idx]
        
        # Create pseudo-label using vegetation indices
        mask = self.create_vegetation_mask(image)
        
        # Convert to tensors
        image_tensor = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        mask_tensor = torch.LongTensor(mask)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor
        }
    
    def create_vegetation_mask(self, image: np.ndarray) -> np.ndarray:
        """Create vegetation mask using color-based indices"""
        # Normalize to [0, 1]
        img = image.astype(np.float32) / 255.0
        
        # Extract RGB channels
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        # Calculate vegetation indices
        # Excess Green Index (ExG)
        exg = 2 * g - r - b
        
        # Visible Atmospherically Resistant Index (VARI)
        # Helps distinguish green vegetation
        with np.errstate(divide='ignore', invalid='ignore'):
            vari = (g - r) / (g + r - b + 1e-8)
            vari = np.nan_to_num(vari, 0)
        
        # Green Leaf Index (GLI)
        with np.errstate(divide='ignore', invalid='ignore'):
            gli = (2 * g - r - b) / (2 * g + r + b + 1e-8)
            gli = np.nan_to_num(gli, 0)
        
        # Combine indices
        vegetation_score = (exg + vari + gli) / 3
        
        # Apply adaptive thresholding
        threshold = np.percentile(vegetation_score[vegetation_score > 0], 60)
        mask = (vegetation_score > threshold).astype(np.uint8)
        
        # Clean up small regions
        from scipy import ndimage
        mask = ndimage.median_filter(mask, size=5)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        
        return mask


def download_gsi_tile(x: int, y: int, z: int = 18) -> Optional[np.ndarray]:
    """Download a single GSI tile"""
    url = f"https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
    
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            return np.array(img)
    except Exception as e:
        print(f"Failed to download tile {x},{y}: {e}")
    
    return None


def get_tiles_for_bbox(bbox: Tuple[float, float, float, float], zoom: int = 18) -> List[np.ndarray]:
    """Download all tiles for a bounding box"""
    west, south, east, north = bbox
    
    def latlon_to_tile(lat, lon, zoom):
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - np.log(np.tan(np.radians(lat)) + 
                            1.0 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n)
        return x, y
    
    x_min, y_max = latlon_to_tile(south, west, zoom)
    x_max, y_min = latlon_to_tile(north, east, zoom)
    
    tiles = []
    total = (x_max - x_min + 1) * (y_max - y_min + 1)
    count = 0
    
    print(f"Downloading {total} tiles...")
    
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            tile = download_gsi_tile(x, y, zoom)
            if tile is not None:
                tiles.append(tile)
                count += 1
                print(f"Downloaded {count}/{total} tiles", end='\r')
    
    print(f"\nSuccessfully downloaded {len(tiles)} tiles")
    return tiles


def train_model(tiles: List[np.ndarray], epochs: int = 10, device: str = 'cpu'):
    """Train the U-Net model on the tiles"""
    # Create dataset
    dataset = GSITileDataset(tiles)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = SimpleUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining U-Net model...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def predict_greenspace(model: nn.Module, image: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Predict greenspace mask for an image"""
    model.eval()
    
    # Prepare image
    image_tensor = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.cpu().numpy().squeeze()


def visualize_results(tiles: List[np.ndarray], model: nn.Module, device: str = 'cpu', num_examples: int = 4):
    """Visualize some prediction results"""
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Select random tiles
    indices = np.random.choice(len(tiles), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        tile = tiles[idx]
        
        # Create pseudo-label for comparison
        dataset = GSITileDataset([tile])
        pseudo_mask = dataset.create_vegetation_mask(tile)
        
        # Predict with model
        pred_mask = predict_greenspace(model, tile, device)
        
        # Plot
        axes[i, 0].imshow(tile)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pseudo_mask, cmap='Greens')
        axes[i, 1].set_title("Pseudo-label (Color-based)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='Greens')
        axes[i, 2].set_title("U-Net Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("greenspace_predictions.png", dpi=150, bbox_inches='tight')
    print("\nVisualization saved to greenspace_predictions.png")


def calculate_greenspace_percentage(mask: np.ndarray) -> float:
    """Calculate the percentage of greenspace in a mask"""
    total_pixels = mask.size
    greenspace_pixels = np.sum(mask > 0)
    return (greenspace_pixels / total_pixels) * 100


def main():
    """Main execution function"""
    print("Greenspace Extraction from GSI Seamless Photo Tiles")
    print("=" * 50)
    
    # Configuration
    # Yokohama Station area (adjust as needed)
    yokohama_bbox = (139.615, 35.461, 139.628, 35.471)  # Small area for demo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Download tiles
    tiles = get_tiles_for_bbox(yokohama_bbox, zoom=18)
    
    if not tiles:
        print("No tiles downloaded. Please check your internet connection.")
        return
    
    # Step 2: Train model
    model = train_model(tiles, epochs=5, device=device)
    
    # Step 3: Save model
    torch.save(model.state_dict(), 'greenspace_unet.pth')
    print("\nModel saved to greenspace_unet.pth")
    
    # Step 4: Analyze results
    print("\nAnalyzing greenspace coverage...")
    total_greenspace = []
    
    for tile in tiles:
        mask = predict_greenspace(model, tile, device)
        percentage = calculate_greenspace_percentage(mask)
        total_greenspace.append(percentage)
    
    avg_greenspace = np.mean(total_greenspace)
    print(f"\nAverage greenspace coverage: {avg_greenspace:.2f}%")
    print(f"Min coverage: {np.min(total_greenspace):.2f}%")
    print(f"Max coverage: {np.max(total_greenspace):.2f}%")
    
    # Step 5: Visualize results
    visualize_results(tiles, model, device, num_examples=min(4, len(tiles)))


if __name__ == "__main__":
    # Install required packages
    print("Installing required packages...")
    os.system("pip install torch torchvision numpy matplotlib pillow requests scipy --break-system-packages")
    
    # Run main
    main()
