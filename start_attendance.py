# start_attendance.py
import subprocess
import sys
import os
import psutil

def is_running():
    for process in psutil.process_iter(['cmdline']):
        try:
            if process.info['cmdline'] and "main.py" in process.info['cmdline']:
                return True
        except:
            pass
    return False

def start_attendance():
    if is_running():
        return "Attendance already running!"

    script_path = os.path.join(os.getcwd(), "main.py")
    if not os.path.exists(script_path):
        return "Error: main.py not found!"

    if sys.platform.startswith("win"):
        subprocess.Popen(
            ["python", "main.py"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        subprocess.Popen(["python3", "main.py"])

    return "Attendance Started!"
