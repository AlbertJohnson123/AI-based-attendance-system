# stop_attendance.py
import psutil

def stop_attendance():
    killed = False

    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            if process.info['cmdline'] and "main.py" in process.info['cmdline']:
                process.kill()
                killed = True
        except:
            pass

    if killed:
        return "Attendance Stopped!"
    return "No attendance process running."
