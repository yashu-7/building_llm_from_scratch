import math
import torch
import tiktoken
import numpy as np
import torch.nn as nn

tokens = np.memmap(r'tokens_dir\tokens.bin', dtype=np.int32, mode='r')
tokens = tokens[:100]
# print(tokens)

tokenizer = tiktoken.get_encoding('cl100k_base')
vocab_size = tokenizer.n_vocab
print(vocab_size)

class PositionalEncoding():
    def __init__(self, context_len, embed_dim):
        pe = torch.zeros(context_len, embed_dim)

        position = torch.arange(0, context_len, dtype=torch.float).unsqueeze(1)
        # print(position.shape)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -math.log(10000.0)/embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe_matrix = pe

class TokenEmbedding():
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
    
    def forward(self, x):
        return self.embedding(x)

embeddings = TokenEmbedding(512, vocab_size)

pe = PositionalEncoding(1024, 512)
pe_matrix = pe.pe_matrix
print(pe_matrix.shape)

for i, token_id in enumerate(tokens):
    token_tensor = torch.tensor(token_id, dtype=torch.long)
    token_embedding = embeddings.forward(token_tensor)

    positional_embedding = pe_matrix[i]


    final_embedding = token_embedding + positional_embedding

    print(final_embedding)
    print(final_embedding.shape)
    break