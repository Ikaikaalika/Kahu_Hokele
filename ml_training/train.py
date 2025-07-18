"""
Training script for astronomical location prediction model
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Tuple
from tqdm import tqdm

from data_loader import AstronomicalDataLoader, create_data_batches
from model import LocationPredictor, create_model, haversine_loss, mse_loss
from augmentation import create_augmentation_pipeline

def find_max_batch_size(model, data_loader, start_batch_size=16, max_batch_size=128):
    """Find the maximum batch size that fits in memory"""
    print(f"Testing batch sizes from {start_batch_size} to {max_batch_size}...")
    
    # Get a small sample for testing
    test_indices = list(range(min(100, len(data_loader.samples))))
    
    working_batch_size = start_batch_size
    
    for batch_size in range(start_batch_size, max_batch_size + 1, 8):
        try:
            print(f"Testing batch size {batch_size}...")
            
            # Create a test batch
            test_batch = data_loader.get_batch(test_indices[:batch_size])
            images, features, targets = test_batch
            
            # Try forward pass
            predictions = model(images, features)
            
            # Try loss computation
            loss = haversine_loss(predictions, targets)
            
            # If we get here, this batch size works
            working_batch_size = batch_size
            print(f"✓ Batch size {batch_size} works")
            
        except Exception as e:
            print(f"✗ Batch size {batch_size} failed: {e}")
            break
    
    print(f"Maximum working batch size: {working_batch_size}")
    return working_batch_size

class Trainer:
    """Training manager for the location prediction model"""
    
    def __init__(self, 
                 model: LocationPredictor,
                 data_loader: AstronomicalDataLoader,
                 learning_rate: float = 1e-4,
                 batch_size: int = 16,
                 save_path: str = "model_checkpoints",
                 use_augmentation: bool = True):
        
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        self.use_augmentation = use_augmentation
        
        # Create augmentation pipelines
        self.train_augmentation = create_augmentation_pipeline(training=True) if use_augmentation else None
        self.val_augmentation = create_augmentation_pipeline(training=False)
        
        # Optimizer 
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 25  # Early stopping patience for longer training
        
        # Get train/val split
        self.train_indices, self.val_indices = data_loader.train_test_split()
        
        print(f"Training samples: {len(self.train_indices)}")
        print(f"Validation samples: {len(self.val_indices)}")
        print(f"Data augmentation: {'Enabled' if use_augmentation else 'Disabled'}")
    
    def loss_fn(self, model, images, features, targets):
        """Combined loss function"""
        predictions = model(images, features)
        
        # Use haversine distance as primary loss
        geo_loss = haversine_loss(predictions, targets)
        
        # Add MSE as regularization (helps with gradient flow)
        mse_loss_val = mse_loss(predictions, targets)
        
        # Combined loss
        total_loss = geo_loss + 0.1 * mse_loss_val
        
        return total_loss, predictions
    
    def train_step(self, batch_indices):
        """Single training step with augmentation"""
        # Get batch with training augmentation
        batch_data = self.data_loader.get_batch(batch_indices, self.train_augmentation)
        images, features, targets = batch_data
        
        # Forward and backward pass
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        (loss, predictions), grads = loss_and_grad_fn(self.model, images, features, targets)
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        return loss, predictions
    
    def validate(self) -> Tuple[float, float]:
        """Validation step without augmentation"""
        val_batches = create_data_batches(self.data_loader, self.val_indices, self.batch_size)
        
        total_loss = 0.0
        total_distance_error = 0.0
        num_samples = 0
        
        for batch_indices in val_batches:
            # Get batch without augmentation for validation
            batch_data = self.data_loader.get_batch(batch_indices, self.val_augmentation)
            images, features, targets = batch_data
            
            # Forward pass only
            predictions = self.model(images, features)
            loss = haversine_loss(predictions, targets)
            
            # Calculate average distance error in km
            distance_error = haversine_loss(predictions, targets)
            
            batch_size = images.shape[0]
            total_loss += float(loss) * batch_size
            total_distance_error += float(distance_error) * batch_size
            num_samples += batch_size
        
        avg_loss = total_loss / num_samples
        avg_distance_error = total_distance_error / num_samples
        
        return avg_loss, avg_distance_error
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state,
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = self.save_path / f"checkpoint_epoch_{epoch}.npz"
        mx.savez(str(checkpoint_path), **checkpoint)
        
        # Save best model separately
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            best_path = self.save_path / "best_model.npz"
            mx.savez(str(best_path), **checkpoint)
            print(f"New best model saved with validation loss: {loss:.4f}")
    
    def train(self, num_epochs: int = 50, save_every: int = 5, validate_every: int = 1):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Training samples: {len(self.train_indices)}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Create training batches (shuffle each epoch)
            np.random.shuffle(self.train_indices)
            train_batches = create_data_batches(self.data_loader, self.train_indices, self.batch_size)
            
            # Training
            epoch_loss = 0.0
            num_batches = len(train_batches)
            
            for batch_idx, batch_indices in enumerate(tqdm(train_batches, desc=f"Epoch {epoch+1}/{num_epochs}")):
                loss, predictions = self.train_step(batch_indices)
                epoch_loss += float(loss)
                
                # Print progress
                if batch_idx % 50 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"Batch {batch_idx}/{num_batches}, Loss: {avg_loss:.4f}")
            
            avg_train_loss = epoch_loss / num_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                val_loss, val_distance_error = self.validate()
                self.val_losses.append(val_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs} completed in {time.time() - start_time:.2f}s")
                print(f"Train Loss: {avg_train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f} km")
                print(f"Val Distance Error: {val_distance_error:.4f} km")
                print("-" * 50)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    self.save_checkpoint(epoch + 1, val_loss)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                # Save regular checkpoint
                if (epoch + 1) % save_every == 0:
                    checkpoint_path = self.save_path / f"checkpoint_epoch_{epoch + 1}.json"
                    results = {
                        'epoch': epoch + 1,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'val_distance_error': val_distance_error,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses
                    }
                    with open(checkpoint_path, 'w') as f:
                        json.dump(results, f, indent=2)
        
        print("Training completed!")
        return self.train_losses, self.val_losses

def main():
    """Main training function"""
    # Configuration
    DATASET_PATH = "/Volumes/X9 Pro/astronomical_dataset"
    IMAGE_SIZE = 224
    # Optimize batch size dynamically
    BATCH_SIZE = 8  # Start conservative for attention model
    LEARNING_RATE = 5e-5  # Lower learning rate for better convergence
    NUM_EPOCHS = 100  # Extended training run with attention and augmentation
    
    print("Initializing data loader...")
    # Use 80% of dataset (202,176 * 0.8 = ~161,740 samples)
    max_samples = int(202176 * 0.8)
    data_loader = AstronomicalDataLoader(DATASET_PATH, image_size=IMAGE_SIZE, max_samples=max_samples)
    
    # Print dataset info
    info = data_loader.get_data_info()
    print(f"Dataset info: {info}")
    
    print("Creating model...")
    model = create_model(image_size=IMAGE_SIZE, feature_dim=info['feature_dim'])
    
    print("Setting up trainer with attention and augmentation...")
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        use_augmentation=True
    )
    
    # Optimize batch size for attention model
    print("Finding optimal batch size for attention model...")
    try:
        optimal_batch_size = find_max_batch_size(model, data_loader, start_batch_size=BATCH_SIZE, max_batch_size=24)
        trainer.batch_size = optimal_batch_size
        print(f"Using optimized batch size: {optimal_batch_size}")
    except Exception as e:
        print(f"Batch size optimization failed: {e}")
        print(f"Using conservative batch size: {BATCH_SIZE}")
    
    print(f"Starting enhanced training with:")
    print(f"  - Batch size: {trainer.batch_size}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Attention mechanisms: Enabled")
    print(f"  - Data augmentation: Enabled")
    print(f"  - Early stopping: Enabled (patience=15)")
    
    train_losses, val_losses = trainer.train(num_epochs=NUM_EPOCHS)
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'image_size': IMAGE_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS
        }
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed and results saved!")

if __name__ == "__main__":
    main()