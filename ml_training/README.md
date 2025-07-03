# Astronomical Location Prediction with MLX

This project uses machine learning to predict geographic location from astronomical images and temporal metadata using Apple's MLX framework.

## Dataset

The model trains on a synthetic astronomical dataset containing:
- **202,176 star field images** (1024×1024 pixels)
- **Global coverage**: -66.5° to 66.5° latitude, -180° to 180° longitude
- **Temporal sampling**: Bi-weekly throughout 2024
- **4 viewing directions** per location/time: North, East, South, West

### Dataset Structure
```
astronomical_dataset/
├── lat_-70_to_-60/
│   ├── lon_-180_to_-170/
│   │   ├── week_00/
│   │   │   ├── north/
│   │   │   │   ├── sky_-65.0_-175.0_00_north.jpg
│   │   │   │   └── sky_-65.0_-175.0_00_north.json
│   │   │   ├── east/
│   │   │   ├── south/
│   │   │   └── west/
```

## Model Architecture

### Multi-Modal Input
- **Image Input**: Star field images (224×224 RGB)
- **Feature Input**: Temporal and viewing metadata
  - Day of year (normalized)
  - Hour of observation (normalized) 
  - Viewing azimuth (normalized)
  - Viewing elevation (normalized)

### Network Design
1. **Image Encoder**: ResNet-style CNN with residual blocks
2. **Feature Encoder**: MLP for auxiliary features
3. **Fusion Layer**: Combines image and feature representations
4. **Location Head**: Outputs latitude/longitude coordinates

### Loss Function
- **Primary**: Haversine distance loss (accounts for Earth's curvature)
- **Auxiliary**: MSE loss for gradient stability

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure MLX is properly installed for Apple Silicon
pip install mlx>=0.15.0
```

## Usage

### Training
```bash
cd ml_training
python train.py
```

**Training Configuration:**
- Image size: 224×224
- Batch size: 8 (optimized for M1 Mac)
- Learning rate: 1e-4
- Epochs: 30
- Optimizer: Adam

### Inference
```bash
# Evaluate on test set
python inference.py

# Single prediction example
from inference import LocationPredictor

predictor = LocationPredictor("model_checkpoints/best_model.npz")
result = predictor.predict_from_files("image.jpg", "metadata.json")
```

## Model Performance

The model predicts geographic coordinates with average error typically under **50 km** on validation data.

### Evaluation Metrics
- **Haversine Distance Error**: Primary metric (km)
- **Coordinate MSE**: Secondary metric
- **Regional Accuracy**: Performance by latitude bands

## Files

- `data_loader.py`: Dataset loading and preprocessing
- `model.py`: MLX neural network architecture
- `train.py`: Training script with checkpointing
- `inference.py`: Model evaluation and prediction
- `requirements.txt`: Python dependencies

## Hardware Requirements

- **Recommended**: Apple Silicon Mac (M1/M2/M3)
- **Memory**: 16GB+ RAM recommended for full dataset
- **Storage**: 60GB+ for dataset and model checkpoints

## Key Features

- **MLX Optimization**: Leverages Apple Silicon GPU acceleration
- **Geographic Loss**: Haversine distance for accurate Earth-surface errors
- **Multi-Modal**: Combines visual and temporal information
- **Checkpointing**: Automatic model saving during training
- **Evaluation Tools**: Comprehensive performance analysis

## Example Results

```
Prediction Results:
Predicted Location: 45.123°, -122.456°
Actual Location: 45.167°, -122.401°
Error: 4.87 km
Date/Time: 2024-03-15T22:00:00
Viewing Direction: {'azimuth': 180, 'elevation': 45}
```

## Next Steps

1. **Fine-tuning**: Adjust hyperparameters for better accuracy
2. **Data Augmentation**: Add noise/transformations for robustness
3. **Model Optimization**: Quantization for mobile deployment
4. **iOS Integration**: Export to Core ML for iPhone apps