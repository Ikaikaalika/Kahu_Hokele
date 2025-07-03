"""
Data augmentation for astronomical images
"""
import numpy as np
import mlx.core as mx
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple

class AstronomicalAugmentation:
    """Data augmentation specifically designed for astronomical star field images"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.9, 1.1),
                 noise_std: float = 0.02,
                 blur_prob: float = 0.1,
                 flip_prob: float = 0.5):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations to an astronomical image"""
        # Start with original image
        aug_image = image.copy()
        
        # Random rotation (small angles to preserve star positions)
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            aug_image = aug_image.rotate(angle, fillcolor=(0, 0, 0))
        
        # Random horizontal flip (stars are symmetric)
        if random.random() < self.flip_prob:
            aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if random.random() < self.flip_prob:
            aug_image = aug_image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Brightness adjustment (simulates atmospheric conditions)
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(aug_image)
            aug_image = enhancer.enhance(brightness_factor)
        
        # Contrast adjustment (simulates different sky conditions)
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(aug_image)
            aug_image = enhancer.enhance(contrast_factor)
        
        # Random blur (simulates atmospheric turbulence)
        if random.random() < self.blur_prob:
            aug_image = aug_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        return aug_image
    
    def augment_array(self, image_array: mx.array) -> mx.array:
        """Apply noise augmentation to image array"""
        # Add Gaussian noise (simulates sensor noise)
        if self.noise_std > 0:
            noise = mx.random.normal(shape=image_array.shape, dtype=image_array.dtype) * self.noise_std
            image_array = mx.clip(image_array + noise, 0.0, 1.0)
        
        return image_array
    
    def augment_features(self, features: mx.array) -> mx.array:
        """Apply small perturbations to auxiliary features"""
        # Add small noise to temporal/viewing features (simulates measurement uncertainty)
        noise_scale = 0.01  # Very small noise for features
        noise = mx.random.normal(shape=features.shape, dtype=features.dtype) * noise_scale
        return features + noise

class AdvancedAugmentation:
    """More sophisticated augmentation techniques"""
    
    def __init__(self):
        self.star_occlusion_prob = 0.1
        self.atmosphere_effect_prob = 0.2
    
    def apply_star_occlusion(self, image: Image.Image) -> Image.Image:
        """Simulate partial star occlusion by clouds"""
        if random.random() > self.star_occlusion_prob:
            return image
        
        # Convert to numpy for manipulation
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create random cloud-like occlusion patterns
        num_patches = random.randint(1, 3)
        for _ in range(num_patches):
            # Random elliptical patches
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius_x = random.randint(20, 80)
            radius_y = random.randint(20, 80)
            
            # Create mask
            y, x = np.ogrid[:height, :width]
            mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            
            # Apply dimming effect
            dimming_factor = random.uniform(0.3, 0.7)
            img_array[mask] = (img_array[mask] * dimming_factor).astype(img_array.dtype)
        
        return Image.fromarray(img_array)
    
    def apply_atmosphere_effect(self, image: Image.Image) -> Image.Image:
        """Simulate atmospheric effects like light pollution or haze"""
        if random.random() > self.atmosphere_effect_prob:
            return image
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add gradient light pollution effect
        height, width = img_array.shape[:2]
        
        # Create gradient from one corner
        corner = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
        intensity = random.uniform(0.02, 0.08)
        
        if corner == 'top-left':
            gradient = np.outer(np.linspace(intensity, 0, height), np.linspace(intensity, 0, width))
        elif corner == 'top-right':
            gradient = np.outer(np.linspace(intensity, 0, height), np.linspace(0, intensity, width))
        elif corner == 'bottom-left':
            gradient = np.outer(np.linspace(0, intensity, height), np.linspace(intensity, 0, width))
        else:  # bottom-right
            gradient = np.outer(np.linspace(0, intensity, height), np.linspace(0, intensity, width))
        
        # Apply gradient to all channels
        for c in range(img_array.shape[2]):
            img_array[:, :, c] = np.clip(img_array[:, :, c] + gradient, 0, 1)
        
        return Image.fromarray((img_array * 255).astype(np.uint8))

def create_augmentation_pipeline(training: bool = True) -> AstronomicalAugmentation:
    """Create appropriate augmentation pipeline for training or validation"""
    if training:
        return AstronomicalAugmentation(
            rotation_range=10.0,          # Small rotations to preserve star patterns
            brightness_range=(0.85, 1.15), # Moderate brightness variation
            contrast_range=(0.9, 1.1),     # Subtle contrast changes
            noise_std=0.015,               # Small amount of noise
            blur_prob=0.08,                # Occasional atmospheric blur
            flip_prob=0.5                  # Flips are valid for star fields
        )
    else:
        # No augmentation for validation
        return AstronomicalAugmentation(
            rotation_range=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            noise_std=0.0,
            blur_prob=0.0,
            flip_prob=0.0
        )