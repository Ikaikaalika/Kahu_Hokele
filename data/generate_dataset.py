#!/usr/bin/env python3
"""
Main execution script for astronomical dataset generation

Usage:
    python generate_dataset.py [--resume JOB_ID] [--validate] [--test] [--workers N]
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from config import *
from batch_processor import BatchProcessor
from dataset_organizer import DatasetOrganizer
from progress_tracker import ProgressTracker

def main():
    parser = argparse.ArgumentParser(description='Generate astronomical dataset for iOS location determination')
    parser.add_argument('--resume', type=int, help='Resume generation from job ID')
    parser.add_argument('--validate', action='store_true', help='Validate dataset integrity only')
    parser.add_argument('--test', action='store_true', help='Generate small test dataset (100 samples)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of worker processes')
    parser.add_argument('--memory-limit', type=int, default=MEMORY_LIMIT_GB, help='Memory limit in GB')
    parser.add_argument('--compress', action='store_true', help='Create compressed archive after generation')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files')
    
    args = parser.parse_args()
    
    print("🌟 Astronomical Dataset Generator")
    print("=" * 50)
    print(f"Target location: {DATASET_ROOT}")
    print(f"Total samples: {TOTAL_SAMPLES:,}")
    print(f"Estimated size: {ESTIMATED_SIZE_GB}GB")
    print(f"External drive: {EXTERNAL_DRIVE_PATH}")
    print()
    
    # Initialize components
    processor = BatchProcessor(max_workers=args.workers, memory_limit_gb=args.memory_limit)
    organizer = DatasetOrganizer()
    tracker = ProgressTracker()
    
    try:
        if args.validate:
            # Validation only
            print("🔍 Validating dataset integrity...")
            validation_results = organizer.validate_dataset_integrity()
            
            if validation_results['missing_files'] or validation_results['corrupt_files']:
                print("❌ Dataset validation failed!")
                return 1
            else:
                print("✅ Dataset validation passed!")
                return 0
        
        elif args.test:
            # Generate test dataset
            print("🧪 Generating test dataset (625 samples)...")
            
            # Modify configuration for test - override config values
            import config
            config.LATITUDE_POINTS = 5
            config.LONGITUDE_POINTS = 5
            config.TOTAL_SAMPLES = config.LATITUDE_POINTS * config.LONGITUDE_POINTS * config.SAMPLES_PER_YEAR * len(config.VIEWING_DIRECTIONS)
            
            # Validate setup
            if not processor.validate_setup():
                print("❌ Setup validation failed!")
                return 1
            
            # Process test dataset
            job_id = processor.process_dataset()
            
            print(f"✅ Test dataset generated! Job ID: {job_id}")
            
        elif args.cleanup:
            # Cleanup only
            print("🧹 Cleaning up temporary files...")
            organizer.cleanup_temp_files()
            print("✅ Cleanup completed!")
            return 0
            
        else:
            # Full dataset generation
            if args.resume:
                # Resume existing job
                job_status = tracker.get_job_status(args.resume)
                if not job_status:
                    print(f"❌ Job {args.resume} not found!")
                    return 1
                
                if job_status['status'] != 'running':
                    print(f"❌ Job {args.resume} is not running (status: {job_status['status']})")
                    return 1
                
                print(f"📄 Resuming job {args.resume}")
                print(f"   Progress: {job_status['progress_percent']:.2f}%")
                print(f"   Completed: {job_status['completed_samples']:,}")
                print(f"   Failed: {job_status['failed_samples']:,}")
                print()
                
                job_id = args.resume
                
            else:
                # Check for existing running jobs
                running_jobs = processor.get_resume_candidates()
                if running_jobs:
                    print("⚠️  Found running jobs:")
                    for job_id, started_at, completed, total in running_jobs:
                        progress = (completed / total) * 100
                        print(f"   Job {job_id}: {progress:.1f}% complete (started {started_at})")
                    
                    response = input("Resume existing job? (y/n): ").lower()
                    if response == 'y':
                        job_id = running_jobs[0][0]  # Resume most recent
                    else:
                        print("Starting new job...")
                        job_id = None
                else:
                    job_id = None
            
            # Validate setup
            print("🔧 Validating system setup...")
            if not processor.validate_setup():
                print("❌ Setup validation failed!")
                return 1
            
            # Create directory structure
            print("📁 Creating directory structure...")
            create_directory_structure()
            
            # Process dataset
            print("🚀 Starting dataset generation...")
            start_time = time.time()
            
            job_id = processor.process_dataset(resume_job_id=job_id)
            
            duration_hours = (time.time() - start_time) / 3600
            print(f"⏱️  Generation completed in {duration_hours:.2f} hours")
            
            # Generate manifest and documentation
            print("📋 Generating dataset manifest...")
            organizer.generate_dataset_manifest()
            
            print("📖 Generating README...")
            organizer.generate_readme()
            
            # Validate final dataset
            print("🔍 Validating final dataset...")
            validation_results = organizer.validate_dataset_integrity()
            
            if validation_results['missing_files']:
                print(f"⚠️  Warning: {len(validation_results['missing_files'])} missing files")
            
            if validation_results['corrupt_files']:
                print(f"⚠️  Warning: {len(validation_results['corrupt_files'])} corrupt files")
            
            # Cleanup
            print("🧹 Cleaning up temporary files...")
            organizer.cleanup_temp_files()
            
            # Optional compression
            if args.compress:
                print("🗜️  Creating compressed archive...")
                archive_path = organizer.create_compressed_archive()
                print(f"✅ Archive created: {archive_path}")
            
            print(f"✅ Dataset generation completed! Job ID: {job_id}")
            
            # Final statistics
            final_stats = tracker.get_progress_summary()
            print("\n📊 Final Statistics:")
            print(f"   Total samples: {final_stats['total_completed_samples']:,}")
            print(f"   Failed samples: {final_stats['total_failed_samples']:,}")
            print(f"   Success rate: {(final_stats['total_completed_samples'] / TOTAL_SAMPLES) * 100:.2f}%")
            
            if validation_results['size_statistics']:
                stats = validation_results['size_statistics']
                print(f"   Total size: {stats['total_image_size_gb']:.1f}GB")
                print(f"   Average image size: {stats['avg_image_size_kb']:.1f}KB")
    
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

def show_progress_summary():
    """Show current progress summary"""
    tracker = ProgressTracker()
    summary = tracker.get_progress_summary()
    
    print("📊 Current Progress Summary:")
    print(f"   Total jobs: {summary['total_jobs']}")
    print(f"   Running jobs: {summary['running_jobs']}")
    print(f"   Completed jobs: {summary['completed_jobs']}")
    print(f"   Total samples completed: {summary['total_completed_samples']:,}")
    print(f"   Total samples failed: {summary['total_failed_samples']:,}")
    print(f"   Overall progress: {summary['overall_progress_percent']:.2f}%")

def create_sample_subset():
    """Create a small sample subset for testing"""
    organizer = DatasetOrganizer()
    
    if not DATASET_ROOT.exists():
        print("❌ Dataset not found! Generate dataset first.")
        return 1
    
    output_dir = DATASET_ROOT.parent / "sample_subset"
    organizer.generate_sample_subset(output_dir, sample_count=50)
    print(f"✅ Sample subset created in {output_dir}")

if __name__ == "__main__":
    # Handle special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            show_progress_summary()
            sys.exit(0)
        elif sys.argv[1] == "sample":
            create_sample_subset()
            sys.exit(0)
    
    sys.exit(main())