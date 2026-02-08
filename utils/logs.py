import os
import sys


class Logger(object):
    def __init__(self, log_file="logs/attendance_system.log"):
        self.terminal = sys.stdout
        # Ensure the directory exists before opening the file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure it writes to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()