import os
import psutil
import shutil
import logging
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Define the cache directory
cache_dir = os.path.expanduser(r"\path to your cache\.cache\huggingface\hub\datasets--HuggingFaceFW--fineweb-edu")
logger.info(f"Checking cache directory: {cache_dir}")

# Delete old cache
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        logger.info(f"Deleted old cache at {cache_dir}")
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
else:
    logger.info("No existing cache found.")

# Check free disk space
try:
    disk = psutil.disk_usage("C:\\")
    free_gb = disk.free / (1024**3)
    logger.info(f"Free disk space on C:\\: {free_gb:.2f} GB")
    if free_gb < 20:
        logger.warning("Low disk space. Ensure at least 20 GB free for dataset and pagefile.sys.")
except Exception as e:
    logger.error(f"Error checking disk space: {e}")

# Number of samples (~12 GB based on 4.83 KB/sample)
SAMPLES_TO_DOWNLOAD = 2500000
CHUNK_SIZE = 250000  # Process 250k samples per chunk to minimize RAM usage

logger.info(f"Starting download of {SAMPLES_TO_DOWNLOAD:,} samples...")
logger.info("Using streaming mode to limit disk usage.")

# Stream the dataset
try:
    my_10gb_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Collect and save in chunks
output_dir = "fineweb_10gb_dataset"
os.makedirs(output_dir, exist_ok=True)
limited_dataset = []
chunk_index = 0
chunk_dirs = []

try:
    for i, sample in enumerate(my_10gb_dataset):
        if i >= SAMPLES_TO_DOWNLOAD:
            break
        limited_dataset.append(sample)
        
        # Save chunk when reaching CHUNK_SIZE
        if len(limited_dataset) >= CHUNK_SIZE:
            try:
                chunk_dataset = Dataset.from_list(limited_dataset)
                chunk_dir = os.path.join(output_dir, f"chunk_{chunk_index}")
                chunk_dataset.save_to_disk(chunk_dir)
                logger.info(f"Saved chunk {chunk_index} with {len(limited_dataset):,} samples to {chunk_dir}")
                chunk_dirs.append(chunk_dir)
                limited_dataset = []  # Clear memory
                chunk_index += 1
            except Exception as e:
                logger.error(f"Error saving chunk {chunk_index}: {e}")
                raise
        
        if i % 50000 == 0 and i > 0:
            logger.info(f"Processed {i:,} samples...")
except Exception as e:
    logger.error(f"Error processing dataset: {e}")
    raise

# Save any remaining samples
if limited_dataset:
    try:
        chunk_dataset = Dataset.from_list(limited_dataset)
        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_index}")
        chunk_dataset.save_to_disk(chunk_dir)
        logger.info(f"Saved final chunk {chunk_index} with {len(limited_dataset):,} samples to {chunk_dir}")
        chunk_dirs.append(chunk_dir)
    except Exception as e:
        logger.error(f"Error saving final chunk: {e}")
        raise

# Combine chunks into a single dataset
logger.info("Combining chunks...")
try:
    combined_dataset = concatenate_datasets([load_from_disk(chunk_dir) for chunk_dir in chunk_dirs])
    combined_dataset.save_to_disk(output_dir)
    logger.info(f"Combined dataset saved to {output_dir}")
except Exception as e:
    logger.error(f"Error combining datasets: {e}")
    raise

# Clean up chunk directories
for chunk_dir in chunk_dirs:
    try:
        shutil.rmtree(chunk_dir)
        logger.info(f"Deleted temporary chunk directory {chunk_dir}")
    except Exception as e:
        logger.error(f"Error deleting chunk directory {chunk_dir}: {e}")

# Clean up cache files
try:
    num_deleted = combined_dataset.cleanup_cache_files()
    logger.info(f"Cleaned up {num_deleted} cache files.")
except Exception as e:
    logger.error(f"Error cleaning cache files: {e}")

logger.info("\n--- Download Complete ---")
logger.info("Dataset information:")
logger.info(combined_dataset)
logger.info("\nFirst example in your new dataset:")
logger.info(combined_dataset[0])

# Check the size of the saved dataset
if os.path.exists(output_dir):
    try:
        size_bytes = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(output_dir)
            for filename in filenames
        )
        size_gb = size_bytes / (1024**3)
        logger.info(f"Saved dataset size: {size_gb:.2f} GB")
    except Exception as e:
        logger.error(f"Error calculating dataset size: {e}")
