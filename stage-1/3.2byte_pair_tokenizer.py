import os
import tiktoken
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk

dataset_path = 'fineweb_10gb_dataset'

tokenizers = tiktoken.list_encoding_names()
print(tokenizers)

tokenizer = tiktoken.get_encoding('cl100k_base')
print(tokenizer.special_tokens_set)

# SOS_TOKEN = '|<StartofToken>|'
# Redundent in newer models as it is Autoregresive

EOT_TOKEN = '<|endoftext|>'
eot_token_id = tokenizer.encode_single_token(EOT_TOKEN)
print(eot_token_id)

token_dir = r'tokens_dir'
token_file = 'tokens.bin'

os.makedirs(token_dir, exist_ok=True)

output_token_file = os.path.join(token_dir, token_file)

if os.path.exists(dataset_path):
    complete_data = load_from_disk(dataset_path)
    with open(output_token_file, 'wb') as f:
        for data in tqdm(complete_data, desc='Tokenizing and saving'):

            token_ids = tokenizer.encode(data['text'], allowed_special={EOT_TOKEN})
            token_ids.append(eot_token_id)
            
            np.array(token_ids, dtype=np.uint32).tofile(f)

print("Tokenization and saving done loading from memory")
tokens = np.memmap(output_token_file, dtype=np.uint32, mode='r')
print(len(tokens))
