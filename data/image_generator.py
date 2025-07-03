"""
Realistic star field image generation with atmospheric effects
"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import json
from datetime import datetime
from star_catalog import StarCatalog
from config import *

class SkyImageGenerator:
    """Generates realistic star field images"""
    
    def __init__(self):
        self.catalog = StarCatalog()
        self.width = IMAGE_WIDTH
        self.height = IMAGE_HEIGHT
        
    def generate_sky_image(self, latitude, longitude, date_time, azimuth, elevation):
        """Generate a realistic star field image"""
        # Create black background
        image = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Get visible stars
        stars = self.catalog.get_visible_stars(
            latitude, longitude, date_time, azimuth, elevation, MAX_STAR_MAGNITUDE
        )
        
        # Get moon and planets
        moon = self.catalog.get_moon_position(latitude, longitude, date_time)
        planets = self.catalog.get_planets(latitude, longitude, date_time)
        
        # Draw stars
        star_data = []
        for star in stars:
            x, y = self._celestial_to_image_coords(
                star['azimuth'], star['altitude'], azimuth, elevation
            )
            
            if 0 <= x < self.width and 0 <= y < self.height:
                # Calculate star size based on magnitude
                size = self._magnitude_to_size(star['magnitude'])
                color = self._magnitude_to_color(star['magnitude'])
                
                # Apply atmospheric extinction
                atmospheric_mag = star['magnitude'] + ATMOSPHERE_EXTINCTION * (90 - star['altitude']) / 90
                if atmospheric_mag <= MAX_STAR_MAGNITUDE:
                    # Draw star with gaussian blur for realistic appearance
                    self._draw_star(draw, x, y, size, color, atmospheric_mag)
                    
                    star_data.append({
                        'name': star['name'],
                        'x': x,
                        'y': y,
                        'magnitude': star['magnitude'],
                        'atmospheric_magnitude': atmospheric_mag,
                        'azimuth': star['azimuth'],
                        'altitude': star['altitude']
                    })
        
        # Draw moon if visible
        moon_data = None
        if moon['visible']:
            moon_x, moon_y = self._celestial_to_image_coords(
                moon['azimuth'], moon['altitude'], azimuth, elevation
            )
            
            if 0 <= moon_x < self.width and 0 <= moon_y < self.height:
                moon_size = 30  # Moon apparent size
                self._draw_moon(draw, moon_x, moon_y, moon_size, moon['phase'])
                moon_data = {
                    'x': moon_x,
                    'y': moon_y,
                    'phase': moon['phase'],
                    'magnitude': moon['magnitude'],
                    'azimuth': moon['azimuth'],
                    'altitude': moon['altitude']
                }
        
        # Draw planets
        planet_data = []
        for name, planet in planets.items():
            planet_x, planet_y = self._celestial_to_image_coords(
                planet['azimuth'], planet['altitude'], azimuth, elevation
            )
            
            if 0 <= planet_x < self.width and 0 <= planet_y < self.height:
                planet_size = self._magnitude_to_size(planet['magnitude']) * 1.5
                planet_color = self._get_planet_color(name)
                
                self._draw_star(draw, planet_x, planet_y, planet_size, planet_color, planet['magnitude'])
                
                planet_data.append({
                    'name': name,
                    'x': planet_x,
                    'y': planet_y,
                    'magnitude': planet['magnitude'],
                    'azimuth': planet['azimuth'],
                    'altitude': planet['altitude']
                })
        
        # Add atmospheric effects
        image = self._add_atmospheric_effects(image, latitude, date_time)
        
        # Create metadata
        metadata = {
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'datetime': date_time.isoformat(),
            'viewing_direction': {
                'azimuth': azimuth,
                'elevation': elevation
            },
            'stars': star_data,
            'moon': moon_data,
            'planets': planet_data,
            'image_params': {
                'width': self.width,
                'height': self.height,
                'field_of_view': 60,  # degrees
                'magnitude_limit': MAX_STAR_MAGNITUDE
            }
        }
        
        return image, metadata
    
    def _celestial_to_image_coords(self, star_az, star_alt, center_az, center_alt):
        """Convert celestial coordinates to image pixel coordinates"""
        # Field of view in degrees
        fov = 60
        
        # Calculate angular offsets from center
        delta_az = star_az - center_az
        delta_alt = star_alt - center_alt
        
        # Handle azimuth wraparound
        if delta_az > 180:
            delta_az -= 360
        elif delta_az < -180:
            delta_az += 360
        
        # Project to image coordinates (simple gnomonic projection)
        scale = self.width / fov
        x = self.width / 2 + delta_az * scale * np.cos(np.radians(center_alt))
        y = self.height / 2 - delta_alt * scale
        
        return int(x), int(y)
    
    def _magnitude_to_size(self, magnitude):
        """Convert star magnitude to pixel size"""
        # Brighter stars (lower magnitude) are larger
        size = max(1, int(STAR_SIZE_SCALE * (6 - magnitude)))
        return min(size, 8)  # Cap maximum size
    
    def _magnitude_to_color(self, magnitude):
        """Convert magnitude to star color intensity"""
        # Logarithmic scaling for realistic brightness
        intensity = int(255 * np.exp(-0.4 * magnitude))
        intensity = max(20, min(255, intensity))
        
        # Add slight color variation (simplified stellar colors)
        if magnitude < 0:  # Very bright stars
            return (intensity, intensity, min(255, int(intensity * 1.1)))
        elif magnitude < 2:  # Bright stars
            return (intensity, intensity, intensity)
        else:  # Fainter stars
            return (int(intensity * 0.9), intensity, int(intensity * 0.95))
    
    def _draw_star(self, draw, x, y, size, color, magnitude):
        """Draw a star with realistic appearance"""
        # Ensure coordinates and size are integers
        x, y, size = int(x), int(y), int(size)
        
        # Draw main star body
        if size <= 2:
            draw.point((x, y), fill=color)
        else:
            # Draw larger stars as circles with gradual fade
            for r in range(size, 0, -1):
                alpha = 1.0 - (r - 1) / size
                fade_color = tuple(int(c * alpha) for c in color)
                draw.ellipse([x-r, y-r, x+r, y+r], fill=fade_color)
        
        # Add diffraction spikes for bright stars
        if magnitude < 2.0:
            spike_length = int(size * 2)
            spike_color = tuple(int(c * 0.3) for c in color)
            
            # Vertical spike
            draw.line([x, y-spike_length, x, y+spike_length], fill=spike_color, width=1)
            # Horizontal spike
            draw.line([x-spike_length, y, x+spike_length, y], fill=spike_color, width=1)
    
    def _draw_moon(self, draw, x, y, size, phase):
        """Draw moon with correct phase"""
        # Ensure coordinates are integers
        x, y, size = int(x), int(y), int(size)
        
        moon_color = (240, 240, 200)  # Slight yellow tint
        
        # Draw full moon circle
        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=moon_color)
        
        # Draw phase shadow if not full
        if phase < 0.99:
            shadow_color = (20, 20, 20)
            shadow_width = int(size * (1 - phase))
            if phase < 0.5:  # Waning
                draw.ellipse([x-size//2, y-size//2, x-size//2+shadow_width, y+size//2], fill=shadow_color)
            else:  # Waxing
                draw.ellipse([x+size//2-shadow_width, y-size//2, x+size//2, y+size//2], fill=shadow_color)
    
    def _get_planet_color(self, planet_name):
        """Get characteristic color for planets"""
        colors = {
            'Mercury': (200, 180, 150),
            'Venus': (255, 240, 200),
            'Mars': (255, 180, 120),
            'Jupiter': (240, 220, 180),
            'Saturn': (250, 230, 160)
        }
        return colors.get(planet_name, (255, 255, 255))
    
    def _add_atmospheric_effects(self, image, latitude, date_time):
        """Add subtle atmospheric effects"""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Add subtle sky glow near horizon
        height, width = img_array.shape[:2]
        y_gradient = np.linspace(0, 1, height).reshape(-1, 1)
        
        # Very subtle blue tint near horizon
        horizon_glow = np.zeros_like(img_array)
        horizon_glow[:, :, 2] = (1 - y_gradient) * 5  # Blue channel
        
        # Blend with original
        img_array = np.clip(img_array.astype(np.float32) + horizon_glow, 0, 255).astype(np.uint8)
        
        # Add very subtle noise for realism
        noise = np.random.normal(0, 2, img_array.shape).astype(np.int8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def save_image_and_metadata(self, image, metadata, output_path, filename_base):
        """Save image and metadata to files"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save image
        image_path = output_path / f"{filename_base}.jpg"
        image.save(image_path, 'JPEG', quality=IMAGE_QUALITY, optimize=True)
        
        # Save metadata
        metadata_path = output_path / f"{filename_base}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return image_path, metadata_path