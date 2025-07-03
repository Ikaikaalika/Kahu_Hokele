"""
Configuration file for astronomical dataset generation
"""
import os
from pathlib import Path

# Geographic parameters
LATITUDE_RANGE = (-66.5, 66.5)  # Between polar circles
LONGITUDE_RANGE = (-180, 180)
GRID_SPACING = 5.0  # degrees
LATITUDE_POINTS = int((LATITUDE_RANGE[1] - LATITUDE_RANGE[0]) / GRID_SPACING) + 1  # 27 points
LONGITUDE_POINTS = int((LONGITUDE_RANGE[1] - LONGITUDE_RANGE[0]) / GRID_SPACING)  # 72 points

# Temporal parameters
YEAR = 2024
SAMPLES_PER_YEAR = 26  # Bi-weekly sampling
SAMPLE_INTERVAL_DAYS = 365 / SAMPLES_PER_YEAR  # ~14 days

# Viewing parameters
VIEWING_DIRECTIONS = {
    'north': {'azimuth': 0, 'elevation': 45},
    'east': {'azimuth': 90, 'elevation': 45},
    'south': {'azimuth': 180, 'elevation': 45},
    'west': {'azimuth': 270, 'elevation': 45}
}

# Image parameters
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGE_FORMAT = 'JPEG'
IMAGE_QUALITY = 85

# Star rendering parameters
MIN_STAR_MAGNITUDE = -2.0  # Brightest stars
MAX_STAR_MAGNITUDE = 6.5   # Faintest visible stars
STAR_SIZE_SCALE = 2.0
ATMOSPHERE_EXTINCTION = 0.2  # Atmospheric dimming factor

# File system parameters
EXTERNAL_DRIVE_PATH = Path("/Volumes/X9 Pro")
DATASET_ROOT = EXTERNAL_DRIVE_PATH / "astronomical_dataset"
PROJECT_DATA_ROOT = Path(__file__).parent

# Database parameters
DATABASE_FILE = PROJECT_DATA_ROOT / "generation_progress.db"

# Performance parameters
BATCH_SIZE = 100  # Images to process in memory at once

def get_optimal_workers():
    """Dynamically determine optimal worker count based on CPU cores"""
    import os
    cpu_count = os.cpu_count() or 4
    # Use 75% of available cores, minimum 1, maximum 12
    return max(1, min(12, int(cpu_count * 0.75)))

def get_memory_limit():
    """Dynamically determine memory limit based on available RAM"""
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    # Use 50% of available memory, minimum 4GB, maximum 16GB
    return max(4, min(16, int(total_memory_gb * 0.5)))

MAX_WORKERS = get_optimal_workers()
MEMORY_LIMIT_GB = get_memory_limit()

# Validation parameters
REQUIRED_FREE_SPACE_GB = 60  # Minimum free space required on external drive

# Dataset statistics (estimated)
TOTAL_LOCATIONS = LATITUDE_POINTS * LONGITUDE_POINTS
TOTAL_SAMPLES = TOTAL_LOCATIONS * SAMPLES_PER_YEAR * len(VIEWING_DIRECTIONS)
ESTIMATED_SIZE_GB = 40  # Conservative estimate

def get_coordinates():
    """Generate all coordinate pairs for the dataset"""
    coordinates = []
    for lat in range(LATITUDE_POINTS):
        latitude = LATITUDE_RANGE[0] + lat * GRID_SPACING
        for lon in range(LONGITUDE_POINTS):
            longitude = LONGITUDE_RANGE[0] + lon * GRID_SPACING
            coordinates.append((latitude, longitude))
    return coordinates

def get_week_dates():
    """Generate all sampling dates for 2024"""
    from datetime import datetime, timedelta
    
    start_date = datetime(YEAR, 1, 1)
    dates = []
    
    for week in range(SAMPLES_PER_YEAR):
        date = start_date + timedelta(days=week * SAMPLE_INTERVAL_DAYS)
        dates.append(date)
    
    return dates

def validate_external_drive():
    """Validate external drive availability and space with error handling"""
    import shutil
    
    try:
        if not EXTERNAL_DRIVE_PATH.exists():
            raise FileNotFoundError(f"External drive not found at {EXTERNAL_DRIVE_PATH}")
        
        free_space_gb = shutil.disk_usage(EXTERNAL_DRIVE_PATH).free / (1024**3)
        if free_space_gb < REQUIRED_FREE_SPACE_GB:
            raise RuntimeError(f"Insufficient space on external drive. Required: {REQUIRED_FREE_SPACE_GB}GB, Available: {free_space_gb:.1f}GB")
        
        return free_space_gb
    except OSError as e:
        raise RuntimeError(f"Error accessing external drive: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error validating external drive: {e}")

def get_output_path(latitude, longitude, week, direction):
    """Generate output path for a specific sample"""
    lat_base = int(latitude//10) * 10
    lat_end = lat_base + 10
    lon_base = int(longitude//10) * 10
    lon_end = lon_base + 10
    
    lat_range = f"lat_{lat_base:+03d}_to_{lat_end:+03d}"
    lon_range = f"lon_{lon_base:+03d}_to_{lon_end:+03d}"
    week_str = f"week_{week:02d}"
    
    path = DATASET_ROOT / lat_range / lon_range / week_str / direction
    return path

def create_directory_structure():
    """Create the complete directory structure with error handling"""
    try:
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        
        # Create regional directories
        coordinates = get_coordinates()
        for lat, lon in coordinates:
            for week in range(SAMPLES_PER_YEAR):
                for direction in VIEWING_DIRECTIONS.keys():
                    path = get_output_path(lat, lon, week, direction)
                    path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating directory structure: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error creating directories: {e}")

