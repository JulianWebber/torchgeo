"""
Configuration file for Greenspace Extraction System
"""

# GSI Seamless Photo Configuration
GSI_CONFIG = {
    'base_url': 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto',
    'default_zoom': 18,  # Zoom level 18 provides ~10-20cm resolution
    'tile_size': 256,
    'timeout': 10,
    'max_retries': 3
}

# Area of Interest Examples (West, South, East, North in WGS84)
# Format: (west_longitude, south_latitude, east_longitude, north_latitude)

AREAS = {
    # Japan - Yokohama (GSI Seamless Photo available, very high resolution ~10cm/pixel)
    'yokohama_station': (139.615, 35.461, 139.628, 35.471),
    'yokohama_park': (139.628, 35.443, 139.643, 35.454),
    'minato_mirai': (139.625, 35.451, 139.645, 35.465),
    'yamashita_park': (139.645, 35.442, 139.655, 35.448),

    # Netherlands - Amsterdam (high resolution satellite imagery available)
    'amsterdam_vondelpark': (4.860, 52.357, 4.875, 52.365),  # Vondelpark area
    'amsterdam_center': (4.885, 52.368, 4.900, 52.378),  # Central Amsterdam
    'amsterdam_westerpark': (4.870, 52.380, 4.885, 52.390),  # Westerpark area
    'amsterdam_oost': (4.920, 52.355, 4.935, 52.365),  # Amsterdam Oost with parks
    'amsterdam_jordaan': (4.875, 52.370, 4.890, 52.380),  # Jordaan neighborhood

    # USA - New York City
    'nyc_central_park_south': (-73.980, 40.765, -73.965, 40.775),  # Central Park south
    'nyc_central_park_north': (-73.970, 40.780, -73.955, 40.790),  # Central Park north
    'nyc_brooklyn_prospect': (-73.975, 40.655, -73.960, 40.665),  # Prospect Park

    # UK - London
    'london_hyde_park': (-0.175, 51.503, -0.160, 51.513),  # Hyde Park
    'london_regents_park': (-0.160, 51.525, -0.145, 51.535),  # Regent's Park
    'london_greenwich_park': (-0.005, 51.475, 0.010, 51.485),  # Greenwich Park

    # Germany - Berlin
    'berlin_tiergarten': (13.355, 52.510, 13.370, 52.520),  # Tiergarten
    'berlin_tempelhofer_feld': (13.390, 52.470, 13.410, 52.485),  # Tempelhof Field

    # France - Paris
    'paris_luxembourg_gardens': (2.330, 48.843, 2.345, 48.853),  # Luxembourg Gardens
    'paris_tuileries': (2.320, 48.860, 2.335, 48.868),  # Tuileries Garden

    # Singapore
    'singapore_gardens_by_bay': (103.860, 1.278, 103.875, 1.288),  # Gardens by the Bay

    # Australia - Sydney
    'sydney_royal_botanic': (151.210, -33.870, 151.225, -33.860),  # Royal Botanic Gardens

    # Custom area (user can define)
    'custom': None
}

# Resolution information by region
# Note: Only GSI Seamless Photo (Japan) provides very high resolution at zoom 18 (~10cm/pixel)
# For other regions, you'll need to use alternative satellite imagery sources
RESOLUTION_NOTES = {
    'gsi_seamless': {
        'regions': ['Japan'],
        'zoom_18': '10-20cm per pixel',
        'url': 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg',
        'note': 'Works only for Japan. Use this for highest resolution.'
    },
    'alternative_sources': {
        'google_satellite': 'Global coverage, 10-50cm resolution, requires API key',
        'bing_satellite': 'Global coverage, ~30cm resolution, requires API key',
        'mapbox_satellite': 'Global coverage, 30-50cm resolution, requires API key',
        'sentinel2': 'Global coverage, 10m resolution, free and open',
        'planet_labs': 'Global coverage, 3-5m resolution, commercial'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'UNet',
    'input_channels': 3,  # RGB
    'output_channels': 2,  # Binary classification (greenspace/non-greenspace)
    'features': [64, 128, 256, 512],  # Feature maps at each level
    'dropout': 0.0,
    'batch_norm': True
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 4,
    'epochs': 20,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adam',
    'loss_function': 'cross_entropy',
    'validation_split': 0.2,
    'early_stopping_patience': 5
}

# Vegetation Index Thresholds
VEGETATION_INDICES = {
    'exg_weight': 1.0,  # Excess Green Index
    'vari_weight': 1.0,  # Visible Atmospherically Resistant Index
    'gli_weight': 1.0,   # Green Leaf Index
    'threshold_percentile': 60,  # Percentile for adaptive thresholding
    'min_vegetation_size': 25,   # Minimum size of vegetation patches in pixels
}

# Data Augmentation (if needed)
AUGMENTATION_CONFIG = {
    'random_flip': True,
    'random_rotation': 15,  # degrees
    'random_brightness': 0.1,
    'random_contrast': 0.1,
    'normalize': True
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_predictions': True,
    'save_format': 'png',  # Options: 'png', 'tif', 'both'
    'output_directory': './output',
    'save_intermediate': False,
    'visualization_dpi': 150
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
    'num_workers': 4,
    'pin_memory': True,
    'mixed_precision': False
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_file': 'greenspace_extraction.log',
    'console_output': True
}
