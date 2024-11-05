#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

# Set the Hugging Face HTTP timeout
os.environ["HUGGINGFACE_HUB_HTTP_TIMEOUT"] = "60"  # Set to 60 seconds or more

# Argument parser for data path, regex, and log file
parser = argparse.ArgumentParser(description="Download filtered dataset files from Hugging Face Hub.")
parser.add_argument('--data_path', type=str, required=True, help="Path to download the data.")
parser.add_argument('--regex', type=str, required=True, help="Regex pattern to filter files for download.")
parser.add_argument('--log_file', type=str, required=True, help="Path to log file.")

args = parser.parse_args()
DATA_PATH = args.data_path
REGEX = args.regex
LOG_FILE = args.log_file

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logging
)

# Initialize the Hugging Face API
api = HfApi()

class TqdmToLogger(tqdm):
    """Custom TQDM progress bar that logs to a specified logger instead of printing to console."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("progress_bar")

    def display(self, msg=None, pos=None):
        self.logger.info(msg or self.format_meter(self.n, self.total, self.elapsed))

def main():
    logging.info("Starting the download process.")

    # Download files matching the regex pattern using the Hugging Face Hub API
    download_path = snapshot_download(
        repo_id="LEAP/ClimSim_low-res",
        allow_patterns=REGEX,
        cache_dir=DATA_PATH,
        repo_type="dataset",
        # tqdm_class=TqdmToLogger  # Use custom tqdm logger for progress tracking
    )

    logging.info(f"Downloaded the dataset to {download_path}")

if __name__ == "__main__":
    main()
