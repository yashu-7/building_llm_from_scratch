from datasets import load_dataset
import sys

# --- Configuration ---
DATASET_ID = "HuggingFaceFW/fineweb-edu"
TARGET_SIZE_GB = 10
NUM_SAMPLES_TO_ESTIMATE = 1000 # Use 1000 samples for a good average

# --- Calculation ---
# 1. Calculate target size in bytes
target_size_bytes = TARGET_SIZE_GB * (1024**3)

# 2. Stream a small number of samples to estimate the average size
print(f"Streaming {NUM_SAMPLES_TO_ESTIMATE} samples to estimate average size...")
streaming_dataset = load_dataset(DATASET_ID, split="train", streaming=True)

total_text_bytes = 0
samples_counted = 0
for sample in streaming_dataset.take(NUM_SAMPLES_TO_ESTIMATE):
    # The 'text' field is the main contributor to the size
    total_text_bytes += sys.getsizeof(sample['text'])
    samples_counted += 1

# 3. Calculate the average size and estimate the total number of samples needed
average_sample_bytes = total_text_bytes / samples_counted
estimated_samples_for_target = int(target_size_bytes / average_sample_bytes)

# --- Output the results ---
print("\n--- Estimation Complete ---")
print(f"Average size per sample: {average_sample_bytes:.2f} bytes")
print(f"To get {TARGET_SIZE_GB}GB of data, you need to download approximately {estimated_samples_for_target:,} samples.")
print("---------------------------\n")

print(f"Use the number {estimated_samples_for_target} in the next step to perform the actual download.")
