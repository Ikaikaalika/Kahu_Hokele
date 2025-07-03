"""
Star catalog management and astronomical calculations
"""
import ephem
import numpy as np
from datetime import datetime, timezone
import sqlite3
from pathlib import Path
from config import PROJECT_DATA_ROOT

class StarCatalog:
    """Manages star catalog and astronomical calculations"""
    
    def __init__(self):
        self.catalog_db = PROJECT_DATA_ROOT / "star_catalog.db"
        self.init_database()
        self.load_bright_stars()
    
    def init_database(self):
        """Initialize star catalog database with error handling"""
        try:
            conn = sqlite3.connect(self.catalog_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stars (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    ra REAL,
                    dec REAL,
                    magnitude REAL,
                    spectral_type TEXT,
                    catalog_id TEXT
                )
            ''')
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def load_bright_stars(self):
        """Load bright star catalog using PyEphem with error handling"""
        try:
            conn = sqlite3.connect(self.catalog_db)
            cursor = conn.cursor()
            
            # Check if stars are already loaded
            cursor.execute("SELECT COUNT(*) FROM stars")
            if cursor.fetchone()[0] > 0:
                return
        except sqlite3.Error as e:
            print(f"Database error checking star count: {e}")
            if conn:
                conn.close()
            raise
        
        # Bright stars from Hipparcos catalog (simplified list)
        bright_stars = [
            ("Sirius", "06:45:08.9", "-16:42:58", -1.46, "A1V"),
            ("Canopus", "06:23:57.1", "-52:41:44", -0.74, "A9II"),
            ("Arcturus", "14:15:39.7", "+19:10:57", -0.05, "K1.5III"),
            ("Vega", "18:36:56.3", "+38:47:01", 0.03, "A0V"),
            ("Capella", "05:16:41.4", "+45:59:53", 0.08, "G5III"),
            ("Rigel", "05:14:32.3", "-08:12:06", 0.13, "B8Ia"),
            ("Procyon", "07:39:18.1", "+05:13:30", 0.34, "F5IV"),
            ("Betelgeuse", "05:55:10.3", "+07:24:25", 0.50, "M1Ia"),
            ("Achernar", "01:37:42.8", "-57:14:12", 0.46, "B6Vep"),
            ("Hadar", "14:03:49.4", "-60:22:23", 0.61, "B1III"),
            ("Altair", "19:50:47.0", "+08:52:06", 0.77, "A7V"),
            ("Aldebaran", "04:35:55.2", "+16:30:33", 0.85, "K5III"),
            ("Antares", "16:29:24.5", "-26:25:55", 1.09, "M1.5Iab"),
            ("Spica", "13:25:11.6", "-11:09:41", 1.04, "B1III"),
            ("Pollux", "07:45:18.9", "+28:01:34", 1.14, "K0III"),
            ("Fomalhaut", "22:57:39.0", "-29:37:20", 1.16, "A3V"),
            ("Deneb", "20:41:25.9", "+45:16:49", 1.25, "A2Ia"),
            ("Mimosa", "12:47:43.3", "-59:41:19", 1.25, "B0.5III"),
            ("Regulus", "10:08:22.3", "+11:58:02", 1.35, "B7V"),
            ("Adhara", "06:58:37.5", "-28:58:20", 1.50, "B2II"),
        ]
        
        # Add more stars for realistic star field
        for i, (name, ra_str, dec_str, mag, spec_type) in enumerate(bright_stars):
            # Convert RA/Dec strings to decimal degrees using astropy for accuracy
            from astropy.coordinates import SkyCoord
            from astropy import units as u
            
            try:
                coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                ra_deg = coord.ra.degree
                dec_deg = coord.dec.degree
            except Exception:
                # Fallback to manual parsing if astropy fails
                ra_parts = ra_str.split(':')
                dec_parts = dec_str.split(':')
                
                ra_deg = (float(ra_parts[0]) + float(ra_parts[1])/60 + float(ra_parts[2])/3600) * 15
                dec_deg = float(dec_parts[0]) + float(dec_parts[1])/60 + float(dec_parts[2])/3600
                if dec_str.startswith('-'):
                    dec_deg = -abs(dec_deg)
            
            cursor.execute('''
                INSERT INTO stars (name, ra, dec, magnitude, spectral_type, catalog_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, ra_deg, dec_deg, mag, spec_type, f"HIP_{i+1}"))
        
        # Generate additional fainter stars for realistic star field
        # Use time-based seed for diversity while maintaining some reproducibility
        import time
        np.random.seed(int(time.time()) % 10000)
        n_faint_stars = 2000
        
        for i in range(n_faint_stars):
            ra = np.random.uniform(0, 360)
            dec = np.arcsin(np.random.uniform(-1, 1)) * 180 / np.pi
            mag = np.random.uniform(2.0, 6.5)
            
            cursor.execute('''
                INSERT INTO stars (name, ra, dec, magnitude, spectral_type, catalog_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (f"Star_{i}", ra, dec, mag, "G2V", f"GEN_{i}"))
        
            conn.commit()
            print(f"Loaded {len(bright_stars) + n_faint_stars} stars into catalog")
        except sqlite3.Error as e:
            print(f"Database error loading stars: {e}")
            raise
        except Exception as e:
            print(f"Error loading star catalog: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_visible_stars(self, latitude, longitude, date_time, azimuth, elevation, max_magnitude=6.5):
        """Get stars visible from a specific location and time (optimized)"""
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date_time.strftime('%Y/%m/%d %H:%M:%S')
        
        # Get all stars from database with error handling
        try:
            conn = sqlite3.connect(self.catalog_db)
            cursor = conn.cursor()
            cursor.execute("SELECT name, ra, dec, magnitude FROM stars WHERE magnitude <= ?", (max_magnitude,))
            stars = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error retrieving stars: {e}")
            return []
        finally:
            if conn:
                conn.close()
        
        visible_stars = []
        
        # Pre-compute observer position for batch processing
        obs_lat_rad = np.radians(latitude)
        obs_lon_rad = np.radians(longitude)
        
        # Batch process stars for better performance
        for name, ra, dec, magnitude in stars:
            # Create star object
            star = ephem.FixedBody()
            star._ra = ra * np.pi / 180
            star._dec = dec * np.pi / 180
            star._epoch = ephem.J2000
            
            # Compute position
            star.compute(observer)
            
            # Check if star is above horizon (optimized check)
            if star.alt > 0:
                # Convert to degrees
                star_az = float(star.az) * 180 / np.pi
                star_alt = float(star.alt) * 180 / np.pi
                
                # Quick angular distance check before expensive calculation
                rough_dist = abs(star_az - azimuth) + abs(star_alt - elevation)
                if rough_dist <= 120:  # Quick filter before precise calculation
                    angular_dist = self._angular_distance(azimuth, elevation, star_az, star_alt)
                    
                    # Include stars within field of view (60 degrees)
                    if angular_dist <= 60:
                        visible_stars.append({
                            'name': name,
                            'azimuth': star_az,
                            'altitude': star_alt,
                            'magnitude': magnitude,
                            'angular_distance': angular_dist
                        })
        
        return visible_stars
    
    def _angular_distance(self, az1, alt1, az2, alt2):
        """Calculate angular distance between two points on the celestial sphere"""
        az1, alt1, az2, alt2 = map(np.radians, [az1, alt1, az2, alt2])
        
        return np.arccos(
            np.sin(alt1) * np.sin(alt2) +
            np.cos(alt1) * np.cos(alt2) * np.cos(az2 - az1)
        ) * 180 / np.pi
    
    def get_moon_position(self, latitude, longitude, date_time):
        """Get moon position and phase"""
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date_time.strftime('%Y/%m/%d %H:%M:%S')
        
        moon = ephem.Moon()
        moon.compute(observer)
        
        return {
            'azimuth': float(moon.az) * 180 / np.pi,
            'altitude': float(moon.alt) * 180 / np.pi,
            'phase': moon.moon_phase,
            'magnitude': moon.mag,
            'visible': moon.alt > 0
        }
    
    def get_planets(self, latitude, longitude, date_time):
        """Get visible planet positions"""
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date_time.strftime('%Y/%m/%d %H:%M:%S')
        
        planets = {}
        planet_bodies = {
            'Mercury': ephem.Mercury(),
            'Venus': ephem.Venus(),
            'Mars': ephem.Mars(),
            'Jupiter': ephem.Jupiter(),
            'Saturn': ephem.Saturn()
        }
        
        for name, planet in planet_bodies.items():
            planet.compute(observer)
            if planet.alt > 0:  # Above horizon
                planets[name] = {
                    'azimuth': float(planet.az) * 180 / np.pi,
                    'altitude': float(planet.alt) * 180 / np.pi,
                    'magnitude': planet.mag
                }
        
        return planets