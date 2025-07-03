"""
MLX-based model for astronomical location prediction
Input: Star field images + temporal/viewing features
Output: Geographic coordinates (latitude, longitude)
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        # MLX Conv2d expects weights in (out_channels, kernel_size, kernel_size, in_channels) format
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
    
    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(channels)
        self.relu = nn.ReLU()
    
    def __call__(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)

class ImageEncoder(nn.Module):
    """CNN encoder for star field images"""
    
    def __init__(self, input_channels: int = 3):
        super().__init__()
        
        # Initial conv layers
        self.conv1 = ConvBlock(input_channels, 64, 7, 2)  # 224x224 -> 112x112
        
        # Residual blocks
        self.conv2 = ConvBlock(64, 128, 3, 2)  # 112x112 -> 56x56
        self.res1 = ResidualBlock(128)
        
        self.conv3 = ConvBlock(128, 256, 3, 2)  # 56x56 -> 28x28
        self.res2 = ResidualBlock(256)
        
        self.conv4 = ConvBlock(256, 512, 3, 2)  # 28x28 -> 14x14
        self.res3 = ResidualBlock(512)
        
        # Output feature dimension
        self.feature_dim = 512
    
    def __call__(self, x):
        # MLX Conv2d expects input in (batch, height, width, channels) format
        # x is already in this format from data loader
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        x = self.res1(x)
        
        x = self.conv3(x)
        x = self.res2(x)
        
        x = self.conv4(x)
        x = self.res3(x)
        
        # Global average pooling - manual implementation
        # x is in (batch, height, width, channels) format
        x = mx.mean(x, axis=(1, 2))  # Average over spatial dimensions (H, W)
        
        # x is now (batch, channels)
        
        return x

class FeatureEncoder(nn.Module):
    """MLP encoder for temporal and viewing features"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        self.feature_dim = output_dim
    
    def __call__(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class LocationPredictor(nn.Module):
    """Complete model for astronomical location prediction"""
    
    def __init__(self, 
                 image_size: int = 224,
                 feature_dim: int = 4,
                 hidden_dim: int = 512):
        super().__init__()
        
        # Encoders
        self.image_encoder = ImageEncoder()
        self.feature_encoder = FeatureEncoder(feature_dim, 64, 128)
        
        # Fusion layer
        combined_dim = self.image_encoder.feature_dim + self.feature_encoder.feature_dim
        
        self.fusion_linear1 = nn.Linear(combined_dim, hidden_dim)
        self.fusion_linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fusion_linear3 = nn.Linear(hidden_dim // 2, 128)
        self.relu = nn.ReLU()
        
        # Output heads
        self.location_head1 = nn.Linear(128, 64)
        self.location_head2 = nn.Linear(64, 2)  # latitude, longitude
    
    def __call__(self, images, features):
        # Encode images
        image_features = self.image_encoder(images)
        
        # Encode auxiliary features
        aux_features = self.feature_encoder(features)
        
        # Combine features
        combined = mx.concatenate([image_features, aux_features], axis=1)
        
        # Fusion
        x = self.relu(self.fusion_linear1(combined))
        x = self.relu(self.fusion_linear2(x))
        fused = self.relu(self.fusion_linear3(x))
        
        # Predict location
        x = self.relu(self.location_head1(fused))
        location = self.location_head2(x)
        
        return location

def haversine_loss(pred_coords: mx.array, true_coords: mx.array) -> mx.array:
    """
    Haversine distance loss function for geographic coordinates
    More appropriate than MSE for lat/lon prediction
    """
    # Convert to radians
    pred_lat, pred_lon = pred_coords[:, 0] * np.pi / 180, pred_coords[:, 1] * np.pi / 180
    true_lat, true_lon = true_coords[:, 0] * np.pi / 180, true_coords[:, 1] * np.pi / 180
    
    # Haversine formula
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon
    
    a = mx.sin(dlat/2)**2 + mx.cos(pred_lat) * mx.cos(true_lat) * mx.sin(dlon/2)**2
    c = 2 * mx.arcsin(mx.sqrt(mx.clip(a, 0, 1)))
    
    # Earth radius in km
    R = 6371.0
    distance = R * c
    
    return mx.mean(distance)

def mse_loss(pred_coords: mx.array, true_coords: mx.array) -> mx.array:
    """Standard MSE loss for coordinates"""
    return mx.mean((pred_coords - true_coords) ** 2)

def create_model(image_size: int = 224, feature_dim: int = 4) -> LocationPredictor:
    """Create and initialize the location prediction model"""
    return LocationPredictor(image_size=image_size, feature_dim=feature_dim)