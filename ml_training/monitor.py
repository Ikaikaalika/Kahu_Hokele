#!/usr/bin/env python3
"""
Training monitor script
"""
import time
import subprocess
import os

def get_process_info(pid):
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,pcpu,pmem,time'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
    except:
        pass
    return None

def check_log_size():
    try:
        return os.path.getsize('training_log.txt')
    except:
        return 0

def main():
    pid = 81990  # Training process PID
    print("Monitoring training process...")
    print("PID    CPU%  MEM%  TIME     LOG_SIZE")
    print("-" * 40)
    
    last_log_size = 0
    
    while True:
        process_info = get_process_info(pid)
        if not process_info:
            print(f"Process {pid} not found - training may have completed")
            break
            
        log_size = check_log_size()
        log_change = "📈" if log_size > last_log_size else "  "
        
        print(f"{process_info} {log_size:8}B {log_change}")
        
        if log_size > last_log_size:
            print("  📝 New log output detected!")
            # Show last few lines if log has content
            if log_size > 0:
                try:
                    with open('training_log.txt', 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"  Latest: {lines[-1].strip()}")
                except:
                    pass
        
        last_log_size = log_size
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main()