"""
Inference script for the trained astronomical location prediction model
"""
import mlx.core as mx
import numpy as np
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
from typing import Tuple, Dict

from model import LocationPredictor as LocationPredictorModel, create_model
from data_loader import AstronomicalDataLoader

class LocationInference:
    """Inference wrapper for the trained model"""
    
    def __init__(self, model_path: str, image_size: int = 224):
        self.image_size = image_size
        self.model = create_model(image_size=image_size)
        
        # Load trained weights
        checkpoint = mx.load(model_path)
        self.model.load_weights(checkpoint)
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image_path: str) -> mx.array:
        """Preprocess a single image for inference"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to numpy array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension and convert to MLX array
        return mx.array(image_array)[None, ...]  # (1, H, W, C)
    
    def extract_features_from_metadata(self, metadata: Dict) -> mx.array:
        """Extract features from metadata (same as training)"""
        # Date/time features
        dt_str = metadata['datetime']
        dt = datetime.fromisoformat(dt_str)
        
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour
        
        # Normalize features
        day_of_year_norm = day_of_year / 365.0
        hour_norm = hour / 24.0
        
        # Viewing direction features
        azimuth = metadata['viewing_direction']['azimuth']
        elevation = metadata['viewing_direction']['elevation']
        
        # Normalize angles
        azimuth_norm = azimuth / 360.0
        elevation_norm = elevation / 90.0
        
        # Create feature vector with batch dimension
        features = mx.array([[
            day_of_year_norm,
            hour_norm,
            azimuth_norm,
            elevation_norm
        ]])
        
        return features
    
    def predict(self, image_path: str, metadata: Dict) -> Tuple[float, float]:
        """Predict location from image and metadata"""
        # Preprocess inputs
        image = self.preprocess_image(image_path)
        features = self.extract_features_from_metadata(metadata)
        
        # Make prediction
        with mx.no_grad():
            prediction = self.model(image, features)
        
        # Convert to numpy and extract coordinates
        coords = np.array(prediction[0])  # Remove batch dimension
        latitude, longitude = float(coords[0]), float(coords[1])
        
        return latitude, longitude
    
    def predict_from_files(self, image_path: str, json_path: str) -> Dict:
        """Predict location from image and JSON metadata files"""
        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        # Make prediction
        pred_lat, pred_lon = self.predict(image_path, metadata)
        
        # Get ground truth
        true_lat = metadata['location']['latitude']
        true_lon = metadata['location']['longitude']
        
        # Calculate error (Haversine distance)
        error_km = self.haversine_distance(
            (true_lat, true_lon),
            (pred_lat, pred_lon)
        )
        
        return {
            'predicted': {'latitude': pred_lat, 'longitude': pred_lon},
            'actual': {'latitude': true_lat, 'longitude': true_lon},
            'error_km': error_km,
            'datetime': metadata['datetime'],
            'viewing_direction': metadata['viewing_direction']
        }
    
    @staticmethod
    def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two coordinates in km"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        return R * c

def evaluate_model(model_path: str, dataset_path: str, num_samples: int = 100):
    """Evaluate model on a subset of the dataset"""
    predictor = LocationInference(model_path)
    data_loader = AstronomicalDataLoader(dataset_path)
    
    # Get random sample of test data
    _, test_indices = data_loader.train_test_split()
    test_sample = np.random.choice(test_indices, min(num_samples, len(test_indices)), replace=False)
    
    results = []
    total_error = 0.0
    
    print(f"Evaluating on {len(test_sample)} samples...")
    
    for idx in test_sample:
        sample = data_loader.samples[idx]
        
        # Get prediction
        result = predictor.predict_from_files(
            sample['image_path'],
            sample['image_path'].replace('.jpg', '.json')
        )
        
        results.append(result)
        total_error += result['error_km']
        
        print(f"Sample {len(results)}: Error = {result['error_km']:.2f} km")
    
    avg_error = total_error / len(results)
    print(f"\nAverage prediction error: {avg_error:.2f} km")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'average_error_km': avg_error,
            'num_samples': len(results),
            'results': results
        }, f, indent=2)
    
    return avg_error, results

def demo_prediction(model_path: str, image_path: str, json_path: str):
    """Demo prediction on a single image"""
    predictor = LocationInference(model_path)
    result = predictor.predict_from_files(image_path, json_path)
    
    print("Prediction Results:")
    print(f"Predicted Location: {result['predicted']['latitude']:.3f}°, {result['predicted']['longitude']:.3f}°")
    print(f"Actual Location: {result['actual']['latitude']:.3f}°, {result['actual']['longitude']:.3f}°")
    print(f"Error: {result['error_km']:.2f} km")
    print(f"Date/Time: {result['datetime']}")
    print(f"Viewing Direction: {result['viewing_direction']}")

if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "model_checkpoints/best_model.npz"
    DATASET_PATH = "/Volumes/X9 Pro/astronomical_dataset"
    
    # Evaluate model
    if Path(MODEL_PATH).exists():
        print("Evaluating model...")
        evaluate_model(MODEL_PATH, DATASET_PATH, num_samples=50)
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using train.py")