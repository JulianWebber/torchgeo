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
AREAS = {
    'yokohama_station': (139.615, 35.461, 139.628, 35.471),
    'yokohama_park': (139.628, 35.443, 139.643, 35.454),
    'minato_mirai': (139.625, 35.451, 139.645, 35.465),
    'yamashita_park': (139.645, 35.442, 139.655, 35.448),
    'custom': None  # User can define custom bbox
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
    'epochs': 150,
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
