# logger_setup.py
"""Configures the application logger."""

import logging

logging.basicConfig(
    level=logging.INFO, # Set desired logging level (e.g., INFO, DEBUG)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create specific logger instance
logger = logging.getLogger('recipe_system')

def get_logger():
    """Returns the configured logger instance."""
    return logger