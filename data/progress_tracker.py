"""
Progress tracking and resume capability with SQLite database
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import psutil
import time
from config import DATABASE_FILE, TOTAL_SAMPLES

class ProgressTracker:
    """Tracks generation progress and enables resume functionality"""
    
    def __init__(self):
        self.db_path = DATABASE_FILE
        self.init_database()
        
    def init_database(self):
        """Initialize progress tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main generation jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT,
                total_samples INTEGER,
                completed_samples INTEGER,
                failed_samples INTEGER,
                estimated_duration_hours REAL,
                actual_duration_hours REAL,
                settings TEXT
            )
        ''')
        
        # Individual sample tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                latitude REAL,
                longitude REAL,
                week INTEGER,
                direction TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                file_path TEXT,
                metadata_path TEXT,
                error_message TEXT,
                processing_time_seconds REAL,
                FOREIGN KEY (job_id) REFERENCES generation_jobs (id)
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                timestamp TIMESTAMP,
                cpu_percent REAL,
                memory_percent REAL,
                memory_gb REAL,
                samples_per_hour REAL,
                estimated_completion TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES generation_jobs (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_job(self, settings=None):
        """Start a new generation job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO generation_jobs 
            (started_at, status, total_samples, completed_samples, failed_samples, settings)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            'running',
            TOTAL_SAMPLES,
            0,
            0,
            json.dumps(settings) if settings else None
        ))
        
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        print(f"Started generation job {job_id}")
        return job_id
    
    def complete_job(self, job_id):
        """Mark job as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get job start time
        cursor.execute("SELECT started_at FROM generation_jobs WHERE id = ?", (job_id,))
        started_at = datetime.fromisoformat(cursor.fetchone()[0])
        
        duration_hours = (datetime.now() - started_at).total_seconds() / 3600
        
        cursor.execute('''
            UPDATE generation_jobs 
            SET completed_at = ?, status = ?, actual_duration_hours = ?
            WHERE id = ?
        ''', (datetime.now(), 'completed', duration_hours, job_id))
        
        conn.commit()
        conn.close()
        
        print(f"Completed generation job {job_id} in {duration_hours:.2f} hours")
    
    def is_sample_completed(self, latitude, longitude, week, direction):
        """Check if a specific sample has been completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status FROM samples 
            WHERE latitude = ? AND longitude = ? AND week = ? AND direction = ?
            ORDER BY id DESC LIMIT 1
        ''', (latitude, longitude, week, direction))
        
        result = cursor.fetchone()
        conn.close()
        
        return result and result[0] == 'completed'
    
    def start_sample(self, job_id, latitude, longitude, week, direction):
        """Mark sample as started"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO samples 
            (job_id, latitude, longitude, week, direction, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, latitude, longitude, week, direction, 'processing', datetime.now()))
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return sample_id
    
    def complete_sample(self, sample_id, file_path, metadata_path, processing_time):
        """Mark sample as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE samples 
            SET status = ?, completed_at = ?, file_path = ?, metadata_path = ?, processing_time_seconds = ?
            WHERE id = ?
        ''', ('completed', datetime.now(), str(file_path), str(metadata_path), processing_time, sample_id))
        
        # Update job progress
        cursor.execute('''
            UPDATE generation_jobs 
            SET completed_samples = completed_samples + 1
            WHERE id = (SELECT job_id FROM samples WHERE id = ?)
        ''', (sample_id,))
        
        conn.commit()
        conn.close()
    
    def fail_sample(self, sample_id, error_message):
        """Mark sample as failed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE samples 
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ?
        ''', ('failed', datetime.now(), error_message, sample_id))
        
        # Update job progress
        cursor.execute('''
            UPDATE generation_jobs 
            SET failed_samples = failed_samples + 1
            WHERE id = (SELECT job_id FROM samples WHERE id = ?)
        ''', (sample_id,))
        
        conn.commit()
        conn.close()
    
    def record_performance(self, job_id):
        """Record current performance metrics"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        
        # Calculate samples per hour
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT started_at, completed_samples FROM generation_jobs WHERE id = ?
        ''', (job_id,))
        
        started_at_str, completed_samples = cursor.fetchone()
        started_at = datetime.fromisoformat(started_at_str)
        
        elapsed_hours = (datetime.now() - started_at).total_seconds() / 3600
        samples_per_hour = completed_samples / elapsed_hours if elapsed_hours > 0 else 0
        
        # Estimate completion time
        remaining_samples = TOTAL_SAMPLES - completed_samples
        hours_remaining = remaining_samples / samples_per_hour if samples_per_hour > 0 else 0
        estimated_completion = datetime.now().timestamp() + (hours_remaining * 3600)
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (job_id, timestamp, cpu_percent, memory_percent, memory_gb, samples_per_hour, estimated_completion)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            datetime.now(),
            cpu_percent,
            memory.percent,
            memory_gb,
            samples_per_hour,
            datetime.fromtimestamp(estimated_completion)
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_gb': memory_gb,
            'samples_per_hour': samples_per_hour,
            'estimated_completion': datetime.fromtimestamp(estimated_completion),
            'progress_percent': (completed_samples / TOTAL_SAMPLES) * 100
        }
    
    def get_job_status(self, job_id):
        """Get current job status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, completed_samples, failed_samples, started_at 
            FROM generation_jobs WHERE id = ?
        ''', (job_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        status, completed, failed, started_at = result
        progress_percent = (completed / TOTAL_SAMPLES) * 100
        
        return {
            'status': status,
            'completed_samples': completed,
            'failed_samples': failed,
            'total_samples': TOTAL_SAMPLES,
            'progress_percent': progress_percent,
            'started_at': started_at
        }
    
    def get_incomplete_samples(self):
        """Get list of samples that need to be processed"""
        from config import get_coordinates, get_week_dates, VIEWING_DIRECTIONS
        
        # Get all possible samples
        all_samples = []
        coordinates = get_coordinates()
        
        for lat, lon in coordinates:
            for week in range(len(get_week_dates())):
                for direction in VIEWING_DIRECTIONS.keys():
                    if not self.is_sample_completed(lat, lon, week, direction):
                        all_samples.append((lat, lon, week, direction))
        
        return all_samples
    
    def cleanup_failed_samples(self, job_id):
        """Clean up failed samples for retry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE samples SET status = 'retry' 
            WHERE job_id = ? AND status = 'failed'
        ''', (job_id,))
        
        cursor.execute('''
            UPDATE generation_jobs 
            SET failed_samples = 0
            WHERE id = ?
        ''', (job_id,))
        
        conn.commit()
        conn.close()
        
        print(f"Marked failed samples for retry in job {job_id}")
    
    def get_progress_summary(self):
        """Get a summary of all generation progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_jobs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running_jobs,
                SUM(completed_samples) as total_completed_samples,
                SUM(failed_samples) as total_failed_samples
            FROM generation_jobs
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_jobs': result[0],
            'completed_jobs': result[1],
            'running_jobs': result[2],
            'total_completed_samples': result[3],
            'total_failed_samples': result[4],
            'overall_progress_percent': (result[3] / TOTAL_SAMPLES) * 100 if result[3] else 0
        }