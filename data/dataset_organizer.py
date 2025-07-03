"""
File organization and metadata generation system
"""
import json
import sqlite3
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from config import *
from progress_tracker import ProgressTracker

class DatasetOrganizer:
    """Organizes and packages the generated dataset"""
    
    def __init__(self):
        self.tracker = ProgressTracker()
        self.dataset_root = DATASET_ROOT
        
    def generate_dataset_manifest(self):
        """Generate comprehensive dataset manifest"""
        manifest = {
            'dataset_info': {
                'name': 'Astronomical Location Dataset',
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'description': 'Synthetic star field images for iOS location determination',
                'total_samples': TOTAL_SAMPLES,
                'estimated_size_gb': ESTIMATED_SIZE_GB
            },
            'geographic_coverage': {
                'latitude_range': LATITUDE_RANGE,
                'longitude_range': LONGITUDE_RANGE,
                'grid_spacing_degrees': GRID_SPACING,
                'total_locations': TOTAL_LOCATIONS,
                'coverage_description': 'Global coverage between polar circles, 5-degree grid'
            },
            'temporal_coverage': {
                'year': YEAR,
                'samples_per_year': SAMPLES_PER_YEAR,
                'sampling_interval_days': SAMPLE_INTERVAL_DAYS,
                'description': 'Bi-weekly sampling throughout 2024'
            },
            'viewing_parameters': {
                'directions': VIEWING_DIRECTIONS,
                'field_of_view_degrees': 60,
                'elevation_degrees': 45,
                'description': 'Four cardinal directions per location/time'
            },
            'image_specifications': {
                'width': IMAGE_WIDTH,
                'height': IMAGE_HEIGHT,
                'format': IMAGE_FORMAT,
                'quality': IMAGE_QUALITY,
                'color_space': 'RGB',
                'magnitude_range': [MIN_STAR_MAGNITUDE, MAX_STAR_MAGNITUDE]
            },
            'astronomical_parameters': {
                'coordinate_system': 'WGS84',
                'epoch': 'J2000',
                'star_catalog': 'Hipparcos + synthetic',
                'atmospheric_effects': True,
                'moon_phases': True,
                'planet_positions': True
            }
        }
        
        # Add generation statistics
        stats = self._gather_generation_statistics()
        manifest['generation_statistics'] = stats
        
        # Add file organization
        manifest['file_organization'] = self._describe_file_organization()
        
        # Save manifest
        manifest_path = self.dataset_root / 'dataset_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Dataset manifest saved to {manifest_path}")
        return manifest
    
    def _gather_generation_statistics(self):
        """Gather statistics from generation process"""
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_jobs,
                SUM(completed_samples) as total_completed,
                SUM(failed_samples) as total_failed,
                AVG(actual_duration_hours) as avg_duration_hours,
                MIN(started_at) as first_started,
                MAX(completed_at) as last_completed
            FROM generation_jobs
            WHERE status = 'completed'
        ''')
        
        job_stats = cursor.fetchone()
        
        # Sample statistics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_samples,
                AVG(processing_time_seconds) as avg_processing_time,
                MIN(processing_time_seconds) as min_processing_time,
                MAX(processing_time_seconds) as max_processing_time
            FROM samples
            WHERE status = 'completed'
        ''')
        
        sample_stats = cursor.fetchone()
        
        # Performance statistics
        cursor.execute('''
            SELECT 
                AVG(cpu_percent) as avg_cpu,
                AVG(memory_percent) as avg_memory,
                MAX(memory_gb) as peak_memory_gb,
                AVG(samples_per_hour) as avg_samples_per_hour
            FROM performance_metrics
        ''')
        
        perf_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'generation_jobs': {
                'total_jobs': job_stats[0] or 0,
                'completed_samples': job_stats[1] or 0,
                'failed_samples': job_stats[2] or 0,
                'average_duration_hours': job_stats[3] or 0,
                'first_started': job_stats[4],
                'last_completed': job_stats[5]
            },
            'sample_processing': {
                'total_processed': sample_stats[0] or 0,
                'average_processing_time_seconds': sample_stats[1] or 0,
                'min_processing_time_seconds': sample_stats[2] or 0,
                'max_processing_time_seconds': sample_stats[3] or 0
            },
            'system_performance': {
                'average_cpu_percent': perf_stats[0] or 0,
                'average_memory_percent': perf_stats[1] or 0,
                'peak_memory_gb': perf_stats[2] or 0,
                'average_samples_per_hour': perf_stats[3] or 0
            }
        }
    
    def _describe_file_organization(self):
        """Describe the file organization structure"""
        return {
            'directory_structure': {
                'root': str(self.dataset_root),
                'pattern': 'lat_range/lon_range/week_XX/direction/',
                'example': 'lat_-10_to_+00/lon_-180_to_-170/week_12/north/',
                'description': 'Hierarchical organization by geography and time'
            },
            'file_naming': {
                'image_pattern': 'sky_LLLL.L_LLLLL.L_WW_DDDD.jpg',
                'metadata_pattern': 'sky_LLLL.L_LLLLL.L_WW_DDDD.json',
                'description': 'L=latitude, L=longitude, W=week, D=direction'
            },
            'metadata_structure': {
                'location': 'WGS84 coordinates',
                'datetime': 'ISO 8601 format',
                'viewing_direction': 'azimuth and elevation in degrees',
                'celestial_objects': 'stars, moon, planets with positions',
                'image_parameters': 'width, height, field of view'
            }
        }
    
    def validate_dataset_integrity(self):
        """Validate dataset integrity and completeness"""
        print("Validating dataset integrity...")
        
        validation_results = {
            'total_expected': TOTAL_SAMPLES,
            'files_found': 0,
            'missing_files': [],
            'corrupt_files': [],
            'metadata_issues': [],
            'size_statistics': {}
        }
        
        # Get all expected samples
        coordinates = get_coordinates()
        week_dates = get_week_dates()
        
        image_sizes = []
        metadata_sizes = []
        
        for lat, lon in coordinates:
            for week in range(len(week_dates)):
                for direction in VIEWING_DIRECTIONS.keys():
                    output_path = get_output_path(lat, lon, week, direction)
                    filename_base = f"sky_{lat:+06.1f}_{lon:+07.1f}_{week:02d}_{direction}"
                    
                    image_path = output_path / f"{filename_base}.jpg"
                    metadata_path = output_path / f"{filename_base}.json"
                    
                    # Check image file
                    if image_path.exists():
                        try:
                            # Basic file size check
                            size = image_path.stat().st_size
                            if size < 1000:  # Too small
                                validation_results['corrupt_files'].append(str(image_path))
                            else:
                                image_sizes.append(size)
                                validation_results['files_found'] += 1
                        except Exception as e:
                            validation_results['corrupt_files'].append(f"{image_path}: {e}")
                    else:
                        validation_results['missing_files'].append(str(image_path))
                    
                    # Check metadata file
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            
                            # Validate metadata structure
                            required_fields = ['location', 'datetime', 'viewing_direction', 'stars']
                            for field in required_fields:
                                if field not in metadata:
                                    validation_results['metadata_issues'].append(
                                        f"{metadata_path}: missing {field}"
                                    )
                            
                            metadata_sizes.append(metadata_path.stat().st_size)
                            
                        except Exception as e:
                            validation_results['metadata_issues'].append(f"{metadata_path}: {e}")
                    else:
                        validation_results['missing_files'].append(str(metadata_path))
        
        # Calculate size statistics
        if image_sizes:
            validation_results['size_statistics'] = {
                'total_images': len(image_sizes),
                'avg_image_size_kb': sum(image_sizes) / len(image_sizes) / 1024,
                'total_image_size_gb': sum(image_sizes) / (1024**3),
                'avg_metadata_size_kb': sum(metadata_sizes) / len(metadata_sizes) / 1024 if metadata_sizes else 0,
                'total_metadata_size_mb': sum(metadata_sizes) / (1024**2) if metadata_sizes else 0
            }
        
        # Save validation report
        validation_path = self.dataset_root / 'validation_report.json'
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"Validation completed:")
        print(f"  Files found: {validation_results['files_found']:,}")
        print(f"  Missing files: {len(validation_results['missing_files']):,}")
        print(f"  Corrupt files: {len(validation_results['corrupt_files']):,}")
        print(f"  Metadata issues: {len(validation_results['metadata_issues']):,}")
        
        if validation_results['size_statistics']:
            stats = validation_results['size_statistics']
            print(f"  Total size: {stats['total_image_size_gb']:.1f}GB")
            print(f"  Avg image size: {stats['avg_image_size_kb']:.1f}KB")
        
        return validation_results
    
    def create_compressed_archive(self, archive_name="astronomical_dataset.zip"):
        """Create compressed archive of the dataset"""
        archive_path = self.dataset_root.parent / archive_name
        
        print(f"Creating compressed archive: {archive_path}")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Add all files in dataset
            total_files = 0
            
            for file_path in self.dataset_root.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.dataset_root)
                    zipf.write(file_path, arcname)
                    total_files += 1
                    
                    if total_files % 1000 == 0:
                        print(f"  Archived {total_files:,} files...")
        
        archive_size_gb = archive_path.stat().st_size / (1024**3)
        
        print(f"Archive created: {archive_path}")
        print(f"  Files: {total_files:,}")
        print(f"  Size: {archive_size_gb:.1f}GB")
        
        return archive_path
    
    def generate_readme(self):
        """Generate comprehensive README for the dataset"""
        readme_content = f"""# Astronomical Location Dataset

## Overview
This dataset contains {TOTAL_SAMPLES:,} synthetic star field images designed for training iOS applications to determine geographic location from night sky photographs.

## Dataset Specifications

### Geographic Coverage
- **Latitude Range**: {LATITUDE_RANGE[0]}° to {LATITUDE_RANGE[1]}° (between polar circles)
- **Longitude Range**: {LONGITUDE_RANGE[0]}° to {LONGITUDE_RANGE[1]}° (global coverage)
- **Grid Spacing**: {GRID_SPACING}° intervals
- **Total Locations**: {TOTAL_LOCATIONS:,} coordinate pairs

### Temporal Coverage
- **Year**: {YEAR}
- **Sampling**: Bi-weekly ({SAMPLES_PER_YEAR} samples per year)
- **Interval**: ~{SAMPLE_INTERVAL_DAYS:.1f} days between samples
- **Coverage**: Full seasonal variation

### Viewing Parameters
- **Directions**: North, East, South, West
- **Elevation**: 45° (optimal viewing angle)
- **Field of View**: 60° per image
- **Total Views**: 4 per location/time

## File Organization

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
│   │   ├── week_01/
│   │   └── ...
│   └── ...
├── dataset_manifest.json
├── validation_report.json
└── README.md
```

## Image Specifications
- **Resolution**: {IMAGE_WIDTH}×{IMAGE_HEIGHT} pixels
- **Format**: JPEG (quality {IMAGE_QUALITY})
- **Color Space**: RGB
- **Content**: Realistic star fields with atmospheric effects

## Metadata Format
Each image has corresponding JSON metadata:
```json
{{
  "location": {{
    "latitude": -65.0,
    "longitude": -175.0
  }},
  "datetime": "2024-01-01T22:00:00",
  "viewing_direction": {{
    "azimuth": 0,
    "elevation": 45
  }},
  "stars": [...],
  "moon": {{...}},
  "planets": [...],
  "image_params": {{...}}
}}
```

## Astronomical Accuracy
- **Star Catalog**: Hipparcos bright stars + synthetic field stars
- **Magnitude Range**: {MIN_STAR_MAGNITUDE} to {MAX_STAR_MAGNITUDE}
- **Coordinate System**: WGS84 / J2000 epoch
- **Effects Included**:
  - Atmospheric extinction
  - Moon phases and position
  - Planet positions (Mercury, Venus, Mars, Jupiter, Saturn)
  - Realistic stellar colors and magnitudes

## Usage Examples

### Loading Images (Python)
```python
from PIL import Image
import json

# Load image and metadata
image = Image.open('sky_-65.0_-175.0_00_north.jpg')
with open('sky_-65.0_-175.0_00_north.json', 'r') as f:
    metadata = json.load(f)

# Extract location
lat = metadata['location']['latitude']
lon = metadata['location']['longitude']
```

### Training Data Format
The dataset is structured for machine learning workflows:
- **Input**: Star field images (1024×1024 RGB)
- **Target**: Geographic coordinates (latitude, longitude)
- **Additional Features**: Time of observation, viewing direction

## Performance Characteristics
- **Total Size**: ~{ESTIMATED_SIZE_GB}GB
- **Average Image Size**: ~25KB (JPEG compressed)
- **Coverage**: All populated regions between polar circles
- **Temporal Resolution**: Bi-weekly sampling captures seasonal variations

## Quality Assurance
- Astronomical calculations verified against PyEphem
- Image generation includes realistic atmospheric effects
- Metadata validated for consistency and completeness
- Full dataset integrity verification included

## License and Attribution
This synthetic dataset was generated for research and development purposes. Please cite appropriately if used in academic work.

## Technical Requirements
- **Generation Platform**: Apple M1 Mac
- **Dependencies**: Python 3.8+, PyEphem, PIL, NumPy
- **Storage**: Minimum 60GB available space
- **Memory**: 8GB+ recommended for processing

## Contact and Support
For questions about the dataset structure or generation methodology, please refer to the generation scripts and documentation included with this dataset.
"""
        
        readme_path = self.dataset_root / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"README generated: {readme_path}")
        return readme_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files and optimize storage"""
        print("Cleaning up temporary files...")
        
        # Remove any .tmp files
        temp_files = list(self.dataset_root.rglob('*.tmp'))
        for temp_file in temp_files:
            temp_file.unlink()
        
        # Remove empty directories
        for dir_path in self.dataset_root.rglob('*'):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                dir_path.rmdir()
        
        print(f"Cleaned up {len(temp_files)} temporary files")
    
    def generate_sample_subset(self, output_dir, sample_count=100):
        """Generate a small sample subset for testing"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating sample subset of {sample_count} images...")
        
        # Get random sample of files
        all_images = list(self.dataset_root.rglob('*.jpg'))
        if len(all_images) < sample_count:
            sample_count = len(all_images)
            
        import random
        random.shuffle(all_images)
        sample_images = all_images[:sample_count]
        
        # Copy files
        for i, image_path in enumerate(sample_images):
            # Copy image
            new_image_path = output_path / f"sample_{i:03d}.jpg"
            shutil.copy2(image_path, new_image_path)
            
            # Copy metadata
            metadata_path = image_path.with_suffix('.json')
            if metadata_path.exists():
                new_metadata_path = output_path / f"sample_{i:03d}.json"
                shutil.copy2(metadata_path, new_metadata_path)
        
        print(f"Sample subset created in {output_path}")
        return output_path