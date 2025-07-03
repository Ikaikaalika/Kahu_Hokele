# Astronomical Dataset Generator - Usage Guide

## Quick Start

### 1. Setup
```bash
cd data
python setup.py
```

### 2. Test Generation (Small Dataset)
```bash
python generate_dataset.py --test
```

### 3. Full Dataset Generation
```bash
python generate_dataset.py
```

## Detailed Usage

### System Requirements
- **Platform**: macOS (optimized for M1/M2)
- **Python**: 3.8 or higher
- **Memory**: 8GB+ recommended
- **Storage**: 60GB+ free space on external drive
- **External Drive**: Must be mounted at `/Volumes/X9 Pro`

### Installation

1. **Clone and setup**:
   ```bash
   cd data
   python setup.py
   ```

2. **Verify installation**:
   ```bash
   python generate_dataset.py --test
   ```

### Command Line Options

```bash
python generate_dataset.py [OPTIONS]
```

**Options:**
- `--resume JOB_ID`: Resume interrupted generation
- `--validate`: Validate dataset integrity only
- `--test`: Generate small test dataset (625 samples)
- `--workers N`: Number of worker processes (default: 8)
- `--memory-limit N`: Memory limit in GB (default: 8)
- `--compress`: Create compressed archive after generation
- `--cleanup`: Clean up temporary files only

**Special Commands:**
- `python generate_dataset.py status`: Show progress summary
- `python generate_dataset.py sample`: Create sample subset for testing

### Dataset Specifications

#### Geographic Coverage
- **Latitude**: -66.5° to +66.5° (between polar circles)
- **Longitude**: -180° to +180° (global coverage)
- **Grid spacing**: 5° intervals
- **Total locations**: 1,944 coordinate pairs

#### Temporal Coverage
- **Year**: 2024
- **Sampling**: Bi-weekly (26 samples per year)
- **Total timepoints**: 26 per location

#### Viewing Parameters
- **Directions**: North, East, South, West (4 per location/time)
- **Elevation**: 45° (optimal viewing angle)
- **Field of view**: 60° per image

#### Dataset Size
- **Total samples**: ~202,000 images
- **Estimated size**: 30-50GB
- **Generation time**: 3-7 days on M1 Mac

### File Organization

```
/Volumes/X9 Pro/astronomical_dataset/
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

### Metadata Format

Each image has corresponding JSON metadata:

```json
{
  "location": {
    "latitude": -65.0,
    "longitude": -175.0
  },
  "datetime": "2024-01-01T22:00:00",
  "viewing_direction": {
    "azimuth": 0,
    "elevation": 45
  },
  "stars": [
    {
      "name": "Sirius",
      "x": 512,
      "y": 300,
      "magnitude": -1.46,
      "azimuth": 180.5,
      "altitude": 25.3
    }
  ],
  "moon": {
    "x": 800,
    "y": 200,
    "phase": 0.75,
    "azimuth": 45.2,
    "altitude": 30.1
  },
  "planets": [
    {
      "name": "Venus",
      "x": 400,
      "y": 600,
      "magnitude": -4.2,
      "azimuth": 270.1,
      "altitude": 15.8
    }
  ],
  "image_params": {
    "width": 1024,
    "height": 1024,
    "field_of_view": 60,
    "magnitude_limit": 6.5
  }
}
```

## Workflow Examples

### Basic Generation
```bash
# Setup (one time)
python setup.py

# Generate test dataset
python generate_dataset.py --test

# Generate full dataset
python generate_dataset.py --workers 6 --memory-limit 10
```

### Resume Interrupted Generation
```bash
# Check for running jobs
python generate_dataset.py status

# Resume specific job
python generate_dataset.py --resume 5
```

### Validation and Cleanup
```bash
# Validate dataset integrity
python generate_dataset.py --validate

# Clean up temporary files
python generate_dataset.py --cleanup

# Create compressed archive
python generate_dataset.py --compress
```

### Performance Tuning

#### M1/M2 Mac Optimization
- Default settings are optimized for Apple Silicon
- Uses 8 worker processes by default
- Memory management prevents thermal throttling
- Batch processing reduces I/O overhead

#### Memory Management
```bash
# Reduce memory usage for 8GB systems
python generate_dataset.py --memory-limit 6 --workers 4

# Increase for 16GB+ systems
python generate_dataset.py --memory-limit 12 --workers 10
```

#### Storage Optimization
- Images saved as JPEG with 85% quality
- Efficient directory structure reduces seek times
- Metadata compressed as JSON
- Optional ZIP compression available

## Troubleshooting

### Common Issues

#### External Drive Not Found
```
Error: External drive not found at /Volumes/X9 Pro
```
**Solution**: 
1. Ensure external drive is connected
2. Check mount point with `ls /Volumes/`
3. Update `EXTERNAL_DRIVE_PATH` in `config.py` if needed

#### Memory Issues
```
Warning: High memory usage persists
```
**Solution**:
```bash
python generate_dataset.py --memory-limit 4 --workers 2
```

#### PyEphem Installation Issues
```
Error: Failed to install pyephem
```
**Solution**:
```bash
# Install with conda (recommended for M1 Macs)
conda install -c conda-forge pyephem

# Or install with pip
pip install pyephem --no-cache-dir
```

#### Generation Stalled
**Check progress**:
```bash
python generate_dataset.py status
```

**Resume generation**:
```bash
python generate_dataset.py --resume [JOB_ID]
```

### Performance Issues

#### Slow Generation
1. **Check thermal throttling**: Monitor Activity Monitor
2. **Reduce workers**: `--workers 4`
3. **Lower memory limit**: `--memory-limit 6`
4. **Check external drive speed**: Use USB 3.0+ or Thunderbolt

#### High Memory Usage
1. **Reduce batch size**: Edit `BATCH_SIZE` in `config.py`
2. **Lower memory limit**: `--memory-limit 4`
3. **Close other applications**

### Validation Failures

#### Missing Files
```bash
# Check specific failures
python generate_dataset.py --validate

# Resume to fill gaps
python generate_dataset.py --resume [JOB_ID]
```

#### Corrupt Images
```bash
# Clean up and regenerate
python generate_dataset.py --cleanup
python generate_dataset.py --resume [JOB_ID]
```

## Advanced Usage

### Custom Configuration

Edit `config.py` for custom parameters:

```python
# Custom geographic range
LATITUDE_RANGE = (30, 50)  # Focus on specific region
GRID_SPACING = 2.0         # Higher resolution

# Custom image parameters
IMAGE_WIDTH = 2048         # Higher resolution
IMAGE_HEIGHT = 2048
MAX_STAR_MAGNITUDE = 7.0   # Include fainter stars

# Custom performance
BATCH_SIZE = 50           # Smaller batches
MAX_WORKERS = 4           # Fewer workers
```

### Batch Processing Scripts

```bash
#!/bin/bash
# Generate dataset in stages

# Stage 1: Test
python generate_dataset.py --test

# Stage 2: Northern hemisphere
# (modify LATITUDE_RANGE in config.py)
python generate_dataset.py

# Stage 3: Southern hemisphere  
# (modify LATITUDE_RANGE in config.py)
python generate_dataset.py

# Stage 4: Validation and packaging
python generate_dataset.py --validate
python generate_dataset.py --compress
```

### Integration with ML Pipelines

```python
# Example data loader
import json
from PIL import Image
from pathlib import Path

def load_astronomical_dataset(dataset_path):
    """Load astronomical dataset for ML training"""
    
    images = []
    labels = []
    
    for image_path in Path(dataset_path).rglob("*.jpg"):
        # Load image
        image = Image.open(image_path)
        images.append(np.array(image))
        
        # Load metadata
        metadata_path = image_path.with_suffix('.json')
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Extract location as label
        lat = metadata['location']['latitude']
        lon = metadata['location']['longitude']
        labels.append([lat, lon])
    
    return np.array(images), np.array(labels)
```

## Support

### Logs and Debugging
- Progress tracked in SQLite database: `generation_progress.db`
- System metrics recorded during generation
- Detailed error messages saved with failed samples

### Performance Monitoring
```bash
# Real-time progress
python generate_dataset.py status

# System resources
top -pid $(pgrep -f generate_dataset.py)
```

### Configuration Files
- `config.py`: Main configuration
- `requirements.txt`: Python dependencies
- `generation_progress.db`: Progress tracking
- `dataset_manifest.json`: Dataset metadata

For additional support or questions, refer to the generated documentation files or check the SQLite database for detailed generation logs.