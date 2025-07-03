#!/usr/bin/env python3
"""
Setup script for astronomical dataset generator
"""
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version OK: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_external_drive():
    """Check for external drive availability"""
    from config import EXTERNAL_DRIVE_PATH, validate_external_drive
    
    try:
        free_space_gb = validate_external_drive()
        print(f"✅ External drive OK: {free_space_gb:.1f}GB available at {EXTERNAL_DRIVE_PATH}")
        return True
    except Exception as e:
        print(f"❌ External drive check failed: {e}")
        print(f"   Expected path: {EXTERNAL_DRIVE_PATH}")
        print("   Please ensure external drive is connected and mounted")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing module imports...")
    
    required_modules = [
        'numpy', 'ephem', 'PIL', 'cv2', 'sqlite3', 
        'psutil', 'tqdm', 'matplotlib', 'scipy'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All modules imported successfully")
        return True

def create_test_image():
    """Create a test image to verify the system works"""
    print("🖼️  Testing image generation...")
    
    try:
        from star_catalog import StarCatalog
        from image_generator import SkyImageGenerator
        from datetime import datetime
        
        # Test star catalog
        catalog = StarCatalog()
        print("   ✅ Star catalog initialized")
        
        # Test image generation
        generator = SkyImageGenerator()
        test_date = datetime(2024, 6, 21, 22, 0, 0)
        
        image, metadata = generator.generate_sky_image(0, 0, test_date, 0, 45)
        print(f"   ✅ Test image generated: {image.size}")
        print(f"   ✅ Metadata created: {len(metadata['stars'])} stars")
        
        # Save test image
        test_output = Path(__file__).parent / "test_output"
        test_output.mkdir(exist_ok=True)
        
        image_path, metadata_path = generator.save_image_and_metadata(
            image, metadata, test_output, "test_image"
        )
        print(f"   ✅ Test files saved: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_system_info():
    """Display system information"""
    import platform
    import psutil
    
    print("💻 System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # Check if M1/M2 Mac
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("   ✅ Apple Silicon Mac detected - optimizations will be applied")

def main():
    """Main setup function"""
    print("🌟 Astronomical Dataset Generator Setup")
    print("=" * 50)
    
    show_system_info()
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Test imports
    if not test_imports():
        return 1
    
    # Check external drive
    if not check_external_drive():
        print("⚠️  External drive not available - some operations may fail")
    
    # Test image generation
    if not create_test_image():
        return 1
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python generate_dataset.py --test")
    print("2. Run: python generate_dataset.py (for full dataset)")
    print("3. Use --help for more options")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())