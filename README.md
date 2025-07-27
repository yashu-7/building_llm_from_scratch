# LLM Data Preparation Pipeline

This repository contains a pipeline for preparing data for training a large language model (LLM). The pipeline consists of three main steps: estimating the number of samples needed for a dataset, downloading and chunking the dataset, and tokenizing the data using byte-pair encoding (BPE). The process is designed to handle large datasets efficiently, ensuring that the data is properly prepared for model training.

**Flowchart**
The following flowchart illustrates the workflow of the scripts in this repository:
```
┌───────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│   1. Data Estimation      │----->│  2. Download & Chunking │----->│    3. Tokenization       │
│ (1_data_gathering.py)     │      │   (2_check_data.py)     │      │ (3.2_byte_pair_tokenizer.py) │
└───────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
           |                                  |                                  |
           v                                  v                                  v
┌───────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│ Estimate # of samples for │      │   `fineweb_10gb_dataset`│      │    `tokens.bin`          │
│ a 10GB dataset            │      │   (Saved to disk)       │      │ (Numerical representation) │
└───────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```
This flowchart shows the sequence of operations:

**Data Estimation**: Calculate the number of samples needed to achieve a 10GB dataset.
**Download & Chunking**: Download the dataset in chunks and save it to disk.
**Tokenization**: Tokenize the dataset using BPE and save the numerical representation.

## Prerequisites
To run the scripts in this repository, you need the following:

Python 3.10+
Required Libraries:

```
datasets
tiktoken
numpy
psutil
tqdm
```

**Disk Space**: At least 20GB of free disk space to accommodate the dataset and temporary files.
**Hugging Face Account**: Required for accessing the FineWeb-Edu dataset. Ensure you have the Hugging Face CLI installed and authenticated:
```
pip install huggingface_hub
huggingface-cli login
```

## Installation

Clone the Repository:
```
git clone <your-repo-url>
cd <your-repo-name>
```

Install Dependencies:
```
pip install -r requirements.txt
```

Create a requirements.txt file with the following content:
```
datasets
tiktoken
numpy
psutil
tqdm
```


## Usage
**Step 1**: Data Estimation

**Script**: *1_data_gathering.py*

**Description**: Estimates the number of samples required to download a 10GB subset of the FineWeb-Edu dataset.
How to Run:
```
# To run the python file
python 1_data_gathering.py
```

**Output**: The script will output the estimated number of samples needed, which should be used in the next step.

**Step 2**: Download & Chunking

**Script**: *2_check_data.py*

**Description**: Downloads the dataset in chunks, manages disk space, and saves the dataset to disk.

**Configuration**: Update the SAMPLES_TO_DOWNLOAD variable in the script with the number obtained from Step 1.
How to Run:
```
# To run the python file
python 2_check_data.py
```

**Output**: The dataset is saved to fineweb_10gb_dataset.

**Step 3**: Tokenization

**Script**: *3.2_byte_pair_tokenizer.py*

**Description**: Tokenizes the dataset using byte-pair encoding (BPE) and saves the tokenized data as a numerical representation.

How to Run:
```
# To run the python file
python 3.2_byte_pair_tokenizer.py
```

**Output**: The tokenized data is saved to tokens_dir/tokens.bin.

## Output

**Data Estimation**: Provides the number of samples needed for a 10GB dataset.

**Download & Chunking**: Saves the dataset to fineweb_10gb_dataset.

**Tokenization**: Produces tokens.bin, a binary file containing the numerical representation of the tokenized dataset.

# References

## This project is inspired by and follows guidance from:

*[Build a Large Language Model (from Scratch) by Sebastian Raschka](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.manning.com%2Fbooks%2Fbuild-a-large-language-model-from-scratch%3Fa_aid%3Draschka%26a_bid%3D4c2437a0%26chan%3Dmm_github)* : Manning Publications

*[Building LLMs from Scratch YouTube playlist by AI Advantage](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)* : YouTube Playlist

## Additional Information

**Disk Space Management**: Ensure sufficient disk space is available, as the dataset and temporary files can be large.
**Handling Large Datasets**: The scripts are designed to handle large datasets by processing data in chunks, minimizing memory usage.
**Troubleshooting**: If you encounter issues with disk space or memory, consider reducing the chunk size in 2_check_data.py.
