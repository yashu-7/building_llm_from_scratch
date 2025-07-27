from datasets import load_from_disk
import os

# 15.69 GB data
# Define the path to your saved dataset
output_dir = "fineweb_10gb_dataset"

# Check if the dataset exists
if os.path.exists(output_dir):
    # Load the dataset from disk
    print(f"Loading dataset from {output_dir}...")
    training_dataset = load_from_disk(output_dir)
    print("Dataset loaded successfully!")

    # Now you can use 'training_dataset' in your training loop
    print("\n--- Dataset Information ---")
    print(training_dataset)

    print("\n--- First Example ---")
    print(training_dataset[0])
else:
    print(f"Dataset directory not found at: {output_dir}")
