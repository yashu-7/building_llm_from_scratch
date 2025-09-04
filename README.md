# Building a Large Language Model From Scratch

This repository documents the end-to-end process of building a large language model (LLM) from the ground up. The project is divided into distinct stages, starting from data acquisition and preparation, moving through the fundamental concepts of attention mechanisms, architecting the transformer model, and culminating in the training process.
Repository Structure

The project is organized into sequential stages, with each stage residing in its own directory:
**/stage-1**: Contains all Python scripts for data gathering, processing, and tokenization.
**/stage-2**: Includes Jupyter notebooks that break down the core attention mechanisms.
**/stage-3**: A Jupyter notebook that constructs the complete Transformer architecture.
**/stage-4**: The final Python script for training the Transformer model.

## Project Workflow
The complete workflow is illustrated below, showing the progression from data preparation to model training.
```
┌───────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────┐
│        STAGE 1            │----->│         STAGE 2          │----->│         STAGE 3           │----->│       STAGE 4        │
│  Data Prep & Tokenization │      │  Attention Mechanisms    │      │ Transformer Architecture  │      │     Model Training   │
└───────────────────────────┘      └──────────────────────────┘      └───────────────────────────┘      └──────────────────────┘
           |                                  |                                 |                                 |
           v                                  v                                 v                                 v
┌───────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐      ┌──────────────────────┐
│      `tokens.bin`         │      │  Self-Attention, Masked  │      │   Complete Transformer    │      │    Trained LLM       │
│   (Tokenized Dataset)     │      │  & Multi-Head Attention  │      │   Model Implementation    │      │      (Final Model)   │
└───────────────────────────┘      └──────────────────────────┘      └───────────────────────────┘      └──────────────────────┘
```
**Stage 1: Data Preparation Pipeline**

This stage contains a pipeline for preparing data for training an LLM. The pipeline consists of three main steps: estimating the number of samples needed for a dataset, downloading and chunking the dataset, and tokenizing the data using *byte-pair encoding (BPE)*. The process is designed to handle large datasets efficiently.
Flowchart (Stage 1)
### The following flowchart illustrates the workflow of the scripts in this stage:
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

### This flowchart shows the sequence of operations:
**Data Estimation**: Calculate the number of samples needed to achieve a 10GB dataset.
**Download & Chunking**: Download the dataset in chunks and save it to disk.
**Tokenization**: Tokenize the dataset using BPE and save the numerical representation.
## Usage (Stage 1)
In this stage, you have all the python codes related to data downloading and processing.
**Step 1**: *Data Estimation*
**Script**: *1_data_gathering.py*
**Description**: Estimates the number of samples required to download a 10GB subset of the FineWeb-Edu dataset.
How to Run:
```
# To run the python file
python 1_data_gathering.py
```
*Output: The script will output the estimated number of samples needed, which should be used in the next step.*

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
**Output*8: The tokenized data is saved to *tokens_dir/tokens.bin*.
### Output (Stage 1)
**Data Estimation**: Provides the number of samples needed for a 10GB dataset.
**Download & Chunking**: Saves the dataset to fineweb_10gb_dataset.
**Tokenization**: Produces *tokens.bin*, a binary file containing the numerical representation of the tokenized dataset.
## Stage 2: Understanding Attention Mechanisms

This stage explores the core components of the Transformer architecture through detailed Jupyter notebooks. The focus is on understanding how attention mechanisms work.
*self-attention.ipynb*: This notebook introduces the concept of self-attention, a mechanism that allows the model to weigh the importance of different words in an input sequence to better understand their relationships.
*masked-attention.ipynb*: This notebook explains masked self-attention, which is crucial in decoder architectures to prevent the model from "cheating" by looking at future tokens during training.
*multihead-attention.ipynb*: This notebook details the multi-head attention mechanism. This approach runs the self-attention process multiple times in parallel, allowing the model to focus on different parts of the input sequence simultaneously and capture more complex relationships.

## Stage 3: Building the Transformer Architecture
In this stage, the concepts from Stage 2 are integrated to build a full Transformer model.
*architecture.ipynb*: This notebook provides a step-by-step implementation of the Transformer architecture, combining the multi-head attention mechanism and feed-forward neural networks. The Transformer is the foundation for most modern large language models.
## Stage 4: Training the Transformer
The final stage focuses on training the implemented Transformer model on the dataset prepared in **Stage 1**.
*training.py*: This script contains the complete code for training the model. It handles data loading, model initialization, the training loop, and saving the final model weights.
Prerequisites
To run the scripts in this repository, you need the following:
Python 3.10+
Required Libraries:
```
datasets
torch
torchinfo
tiktoken
numpy
psutil
tqdm
```
**Disk Space**: At least **20GB** of free disk space to accommodate the dataset and temporary files.
**Hugging Face Account**: Required for accessing the FineWeb-Edu dataset. Ensure you have the Hugging Face CLI installed and authenticated:
```
pip install huggingface_hub
huggingface-cli login
Installation
```
Clone the Repository:
```
git clone https://github.com/yashu-7/building_llm_from_scratch.git
cd https://github.com/yashu-7/building_llm_from_scratch.git
```
Install Dependencies:
Create a requirements.txt file with the following content:
```
datasets
torch
torchinfo
tiktoken
numpy
psutil
tqdm
```
Then, install the dependencies:
```
pip install -r requirements.txt
```
## Additional Information
**Disk Space Management**: Ensure sufficient disk space is available, as the dataset and temporary files can be large.
**Handling Large Datasets**: The scripts are designed to handle large datasets by processing data in chunks, minimizing memory usage.
**Troubleshooting**: If you encounter issues with disk space or memory, consider reducing the chunk size in *2_check_data.py*.

# References

## This project is inspired by and follows guidance from:

*[Build a Large Language Model (from Scratch) by Sebastian Raschka](https://www.google.com/url?sa=E&q=https%3A%2F%2Fwww.manning.com%2Fbooks%2Fbuild-a-large-language-model-from-scratch%3Fa_aid%3Draschka%26a_bid%3D4c2437a0%26chan%3Dmm_github)* : Manning Publications

*[Building LLMs from Scratch YouTube playlist by AI Advantage](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)* : YouTube Playlist
