"""
Data loader for astronomical location prediction dataset
"""
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import mlx.core as mx
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split

class AstronomicalDataLoader:
    """Data loader for astronomical images and location prediction"""
    
    def __init__(self, dataset_path: str, image_size: int = 224, max_samples: int = None):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.max_samples = max_samples
        self.samples = []
        self._load_dataset_index()
    
    def _load_dataset_index(self):
        """Load and index all dataset samples"""
        print("Loading dataset index...")
        
        # Find all JSON metadata files
        json_files = list(self.dataset_path.rglob("*.json"))
        
        # Filter out manifest, validation files, and macOS system files
        json_files = [f for f in json_files if f.name not in ["dataset_manifest.json", "validation_report.json"] 
                     and not f.name.startswith("._")]
        
        for json_file in json_files:
            try:
                # Load metadata
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get corresponding image file
                image_file = json_file.with_suffix('.jpg')
                
                if image_file.exists():
                    self.samples.append({
                        'image_path': str(image_file),
                        'metadata': metadata
                    })
                    
                    # Stop if we've reached max_samples
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _preprocess_image(self, image_path: str) -> mx.array:
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to MLX array and add channel dimension if needed
        return mx.array(image_array)
    
    def _extract_features(self, metadata: Dict) -> Tuple[mx.array, mx.array]:
        """Extract features from metadata"""
        # Location target (lat, lon)
        location = mx.array([
            metadata['location']['latitude'],
            metadata['location']['longitude']
        ])
        
        # Date/time features
        dt_str = metadata['datetime']
        dt = datetime.fromisoformat(dt_str)
        
        # Convert to useful features
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        # Normalize features
        day_of_year_norm = day_of_year / 365.0  # [0, 1]
        hour_norm = hour / 24.0  # [0, 1]
        
        # Viewing direction features
        azimuth = metadata['viewing_direction']['azimuth']
        elevation = metadata['viewing_direction']['elevation']
        
        # Normalize angles
        azimuth_norm = azimuth / 360.0  # [0, 1]
        elevation_norm = elevation / 90.0  # [0, 1]
        
        # Create feature vector
        features = mx.array([
            day_of_year_norm,
            hour_norm,
            azimuth_norm,
            elevation_norm
        ])
        
        return features, location
    
    def get_batch(self, indices: List[int]) -> Tuple[mx.array, mx.array, mx.array]:
        """Get a batch of data"""
        images = []
        features = []
        locations = []
        
        for idx in indices:
            sample = self.samples[idx]
            
            # Load image
            image = self._preprocess_image(sample['image_path'])
            images.append(image)
            
            # Extract features and target
            feat, loc = self._extract_features(sample['metadata'])
            features.append(feat)
            locations.append(loc)
        
        return (
            mx.stack(images),
            mx.stack(features),
            mx.stack(locations)
        )
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[int], List[int]]:
        """Split dataset into train and test indices"""
        indices = list(range(len(self.samples)))
        return train_test_split(indices, test_size=test_size, random_state=random_state)
    
    def get_data_info(self) -> Dict:
        """Get dataset information"""
        if not self.samples:
            return {}
        
        # Analyze location distribution
        latitudes = [s['metadata']['location']['latitude'] for s in self.samples]
        longitudes = [s['metadata']['location']['longitude'] for s in self.samples]
        
        return {
            'num_samples': len(self.samples),
            'latitude_range': (min(latitudes), max(latitudes)),
            'longitude_range': (min(longitudes), max(longitudes)),
            'image_size': self.image_size,
            'feature_dim': 4,  # day_of_year, hour, azimuth, elevation
            'target_dim': 2   # latitude, longitude
        }

def create_data_batches(data_loader: AstronomicalDataLoader, 
                       indices: List[int], 
                       batch_size: int = 32) -> List[Tuple[mx.array, mx.array, mx.array]]:
    """Create batches from dataset indices"""
    batches = []
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = data_loader.get_batch(batch_indices)
        batches.append(batch)
    
    return batches