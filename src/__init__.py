"""Marks src as a Python package and initializes the .env file and logging."""
import os
import time

from dotenv import load_dotenv

from src.my_logging import setup_logging

# Set correct time to log nicely:
os.environ["TZ"] = "Europe/Amsterdam"
time.tzset()

load_dotenv()

logger = setup_logging("vhk")
logger.info(f'Your environment is {os.environ.get("OTAP", "no OTAP, but local")}')
