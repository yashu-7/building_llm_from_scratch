import os
import torch
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


tokenizer = tiktoken.get_encoding('cl100k_base')
tokens_path = os.path.join('tokens_dir', 'tokens.bin')

tokens = np.memmap(tokens_path, dtype=np.int32, mode='r')
print(len(tokens))
print(tokens[:10])
print(tokenizer.decode(tokens[:10]))

class WebDataset(Dataset):
    def __init__(self, tokens, context_length=512, offset=1):
        super().__init__()
        self.tokens = tokens
        self.context_length = context_length
        self.offset = offset

    def __len__(self):
        return len(self.tokens) - (self.context_length + self.offset)
    
    def __getitem__(self, index):
        inputs = torch.tensor(self.tokens[index : index + self.context_length], dtype=torch.long)
        targets = torch.tensor(self.tokens[index + self.offset : index + self.context_length + self.offset], dtype=torch.long)
        
        return inputs, targets

data = WebDataset(tokens)

train_len = int(0.8 * len(data))
valid_len = int(len(data) - train_len)

train_data, valid_data = random_split(data, [train_len, valid_len])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=64)

inputs, targets = next(iter(train_loader))
print(f"{inputs.shape}\n{targets.shape}")
