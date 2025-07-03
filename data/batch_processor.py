"""
M1-optimized batch processing with memory management
"""
import multiprocessing as mp
import concurrent.futures
import psutil
import time
import gc
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import *
from image_generator import SkyImageGenerator
from progress_tracker import ProgressTracker

class BatchProcessor:
    """M1-optimized batch processor for astronomical image generation"""
    
    def __init__(self, max_workers=None, memory_limit_gb=None):
        self.max_workers = max_workers or min(MAX_WORKERS, mp.cpu_count())
        self.memory_limit_gb = memory_limit_gb or MEMORY_LIMIT_GB
        self.tracker = ProgressTracker()
        
        # Performance monitoring
        self.start_time = None
        self.processed_count = 0
        self.last_memory_check = time.time()
        
        print(f"BatchProcessor initialized:")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Memory limit: {self.memory_limit_gb}GB")
        print(f"  Batch size: {BATCH_SIZE}")
    
    def process_dataset(self, resume_job_id=None):
        """Process the complete dataset with M1 optimization"""
        # Start or resume job
        if resume_job_id:
            job_id = resume_job_id
            print(f"Resuming job {job_id}")
        else:
            settings = {
                'latitude_range': LATITUDE_RANGE,
                'longitude_range': LONGITUDE_RANGE,
                'grid_spacing': GRID_SPACING,
                'samples_per_year': SAMPLES_PER_YEAR,
                'viewing_directions': list(VIEWING_DIRECTIONS.keys()),
                'image_params': {
                    'width': IMAGE_WIDTH,
                    'height': IMAGE_HEIGHT,
                    'magnitude_limit': MAX_STAR_MAGNITUDE
                }
            }
            job_id = self.tracker.start_job(settings)
        
        try:
            # Get incomplete samples
            incomplete_samples = self.tracker.get_incomplete_samples()
            total_remaining = len(incomplete_samples)
            
            print(f"Processing {total_remaining:,} remaining samples...")
            
            if total_remaining == 0:
                print("All samples already completed!")
                self.tracker.complete_job(job_id)
                return job_id
            
            self.start_time = time.time()
            
            # Process in batches to manage memory
            with tqdm(total=total_remaining, desc="Generating images") as pbar:
                for batch_start in range(0, total_remaining, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, total_remaining)
                    batch = incomplete_samples[batch_start:batch_end]
                    
                    # Process batch
                    self._process_batch(job_id, batch, pbar)
                    
                    # Memory management
                    self._manage_memory()
                    
                    # Performance monitoring
                    if time.time() - self.last_memory_check > 60:  # Every minute
                        metrics = self.tracker.record_performance(job_id)
                        self._print_progress_update(metrics, pbar)
                        self.last_memory_check = time.time()
                    
                    # Brief pause between batches for thermal management on M1
                    time.sleep(0.1)
            
            self.tracker.complete_job(job_id)
            print(f"\nDataset generation completed! Job ID: {job_id}")
            
        except KeyboardInterrupt:
            print(f"\nGeneration interrupted. Job {job_id} can be resumed later.")
            raise
        except Exception as e:
            print(f"\nError during generation: {e}")
            raise
        
        return job_id
    
    def _process_batch(self, job_id, batch, pbar):
        """Process a batch of samples with multiprocessing"""
        if len(batch) <= 4:  # Small batch - process serially
            for sample in batch:
                self._process_single_sample(job_id, sample)
                pbar.update(1)
        else:  # Large batch - use multiprocessing
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_sample = {
                    executor.submit(process_sample_worker, job_id, sample): sample
                    for sample in batch
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing sample {sample}: {e}")
                        # Mark sample as failed
                        sample_id = self.tracker.start_sample(job_id, *sample)
                        self.tracker.fail_sample(sample_id, str(e))
                        pbar.update(1)
    
    def _process_single_sample(self, job_id, sample):
        """Process a single sample"""
        try:
            latitude, longitude, week, direction = sample
            
            # Start tracking
            sample_id = self.tracker.start_sample(job_id, latitude, longitude, week, direction)
            start_time = time.time()
            
            # Generate sample
            generator = SkyImageGenerator()
            dates = get_week_dates()
            date_time = dates[week]
            
            view_params = VIEWING_DIRECTIONS[direction]
            azimuth = view_params['azimuth']
            elevation = view_params['elevation']
            
            # Generate image and metadata
            image, metadata = generator.generate_sky_image(
                latitude, longitude, date_time, azimuth, elevation
            )
            
            # Save files
            output_path = get_output_path(latitude, longitude, week, direction)
            filename_base = f"sky_{latitude:+06.1f}_{longitude:+07.1f}_{week:02d}_{direction}"
            
            image_path, metadata_path = generator.save_image_and_metadata(
                image, metadata, output_path, filename_base
            )
            
            # Complete tracking
            processing_time = time.time() - start_time
            self.tracker.complete_sample(sample_id, image_path, metadata_path, processing_time)
            
            self.processed_count += 1
            
        except Exception as e:
            # Mark as failed
            sample_id = self.tracker.start_sample(job_id, latitude, longitude, week, direction)
            self.tracker.fail_sample(sample_id, str(e))
            raise
    
    def _manage_memory(self):
        """Aggressive memory management for M1 Macs"""
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        
        if memory_gb > self.memory_limit_gb:
            print(f"Memory usage high ({memory_gb:.1f}GB), forcing cleanup...")
            
            # More aggressive cleanup
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)
            
            # Reduce batch size temporarily if memory is still high
            new_memory = psutil.virtual_memory().used / (1024**3)
            if new_memory > self.memory_limit_gb * 0.9:
                print("Warning: High memory usage persists")
    
    def _print_progress_update(self, metrics, pbar):
        """Print detailed progress update"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        
        pbar.write(f"\n--- Progress Update ---")
        pbar.write(f"Processed: {self.processed_count:,} samples")
        pbar.write(f"Rate: {metrics['samples_per_hour']:.1f} samples/hour")
        pbar.write(f"CPU: {metrics['cpu_percent']:.1f}%")
        pbar.write(f"Memory: {metrics['memory_gb']:.1f}GB ({metrics['memory_percent']:.1f}%)")
        pbar.write(f"Progress: {metrics['progress_percent']:.2f}%")
        pbar.write(f"ETA: {metrics['estimated_completion'].strftime('%Y-%m-%d %H:%M')}")
        pbar.write("")
    
    def validate_setup(self):
        """Validate system setup before processing"""
        # Check external drive
        try:
            free_space_gb = validate_external_drive()
            print(f"External drive OK - {free_space_gb:.1f}GB available")
        except Exception as e:
            print(f"External drive validation failed: {e}")
            return False
        
        # Check memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        if available_gb < 4:
            print(f"Warning: Low available memory ({available_gb:.1f}GB)")
        
        # Test image generation
        try:
            print("Testing image generation...")
            generator = SkyImageGenerator()
            test_date = datetime(2024, 6, 21, 22, 0, 0)  # Summer solstice
            image, metadata = generator.generate_sky_image(0, 0, test_date, 0, 45)
            print(f"Test image generated: {image.size}, {len(metadata['stars'])} stars")
        except Exception as e:
            print(f"Image generation test failed: {e}")
            return False
        
        print("Setup validation completed successfully!")
        return True
    
    def get_resume_candidates(self):
        """Get list of jobs that can be resumed"""
        summary = self.tracker.get_progress_summary()
        
        if summary['running_jobs'] > 0:
            # Find running jobs
            import sqlite3
            conn = sqlite3.connect(self.tracker.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, started_at, completed_samples, total_samples
                FROM generation_jobs 
                WHERE status = 'running'
                ORDER BY started_at DESC
            ''')
            
            running_jobs = cursor.fetchall()
            conn.close()
            
            return running_jobs
        
        return []

def process_sample_worker(job_id, sample):
    """Worker function for multiprocessing"""
    processor = BatchProcessor(max_workers=1)  # Single worker per process
    processor._process_single_sample(job_id, sample)