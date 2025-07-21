"""
@dateL 2025-02-20
@version: 1.0.0
@description: Class for logging the flow of the code based on the user input controlled by the verbosity level.
"""


import logging

class colorFormatter(logging.Formatter):
    """Custom formatter to add ANSI colors to different log levels in the console."""
    
    COLORS = {
        "DEBUG": "\033[35m",        # Cyan
        "INFO": "\033[36m",         # Magenta
        "WARNING": "\033[33m",      # Yellow
        "ERROR": "\033[31m",        # Red
        "CRITICAL": "\033[31m",     # Red (same as ERROR now)
        "RESET": "\033[0m"          # Reset color
    }
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        log_message = super().format(record)
        return f"{log_color}{log_message}{self.COLORS['RESET']}"
        

class ZLogger:
    def __init__(self, log_file="ZLog.log"):
        # Formatting the log output
        log_format = "%(asctime)s - %(levelname)s - %(message)s"

        # Create a logger and set up the level
        self.logger = logging.getLogger("zlogger")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(logging.Formatter(log_format))

        # Console Handler 
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colorFormatter(log_format))

        # Add the necessary handlers to the logger
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)


    # Get the logger 
    def getLogger(self):
        return self.logger

# GLobal logger instance
zlogger = ZLogger()

#Global method to get the logger
def get_logger():
    return zlogger.getLogger()
