import datasets
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)

# custom_cache_dir = ""

# datasets.config.DOWNLOADED_DATASETS_PATH = Path(custom_cache_dir)

# Download the dataset from the Hugging Face Hub
download_path = snapshot_download(
    repo_id="LEAP/ClimSim_low-res",
    allow_patterns="train/0001-02/*0001-02-01-*.nc",
    # cache_dir=custom_cache_dir,
    repo_type="dataset",
)

print(f"Downloaded the dataset to {download_path}")