import os
import time
import torch
import tiktoken
import numpy as np
from math import log
from tqdm import tqdm
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

tok = time.time()

tokens = np.memmap(r'tokens_dir\tokens.bin', dtype=np.uint32, mode='r')
tokens = tokens[:1_000_000]
print(len(tokens))

tokenizer = tiktoken.get_encoding('cl100k_base')
print(tokenizer.special_tokens_set)
vocab = tokenizer.n_vocab
print(f"Vocab size: {vocab}")

print(tokens[:10])
print(tokenizer.decode(tokens[:10]))

class FineWebData(Dataset):
    def __init__(self, tokens, context_len, stride=1):
        super().__init__()
        self.tokens = tokens
        self.context_len = context_len
        self.stride = stride
    
    def __len__(self):
        return max(0, len(self.tokens) - (self.context_len + self.stride))

    def __getitem__(self, idx):
        inputs = np.array(self.tokens[idx : idx + self.context_len], dtype=np.uint32)
        targets = np.array(self.tokens[idx + self.stride : idx + self.context_len + self.stride], dtype=np.uint32)
        return torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()

class TokenEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
    
    def forward(self, x):
        return self.dropout(self.embedding(x))

class PositionalEncoding(nn.Module):
    def __init__(self, context_len, embed_dim, dropout=0.2):
        super().__init__()
        self.context_len = context_len
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(self.context_len, self.embed_dim)
        position = torch.arange(0, self.context_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -log(10000.0) / self.embed_dim)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, context_len, n_heads, dropout=0.2):
        super().__init__()

        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.context_len = context_len
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads

        self.Q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.K = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.V = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len)))
        
        self.out = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = q @ k.transpose(-2, -1)

        scaled_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float, device=x.device))

        masked_scores = scaled_scores.masked_fill(self.mask[:T, :T]==0, float('-inf'))

        attention_weights = F.softmax(masked_scores, dim=-1)
        attention_weights = self.dropout1(attention_weights)

        context_vectors = attention_weights @ v
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(B, T, C)

        context_vecs = self.out(context_vectors)
        context_vecs = self.dropout2(context_vecs)

        return context_vecs

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, context_dim, n_heads, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.n_heads = n_heads

        self.attention = MultiHeadAttention(self.embed_dim, self.context_dim, self.n_heads)
        
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out = self.attention(x)
        norm1_out = self.norm1(x + self.dropout(attn_out))
        ff_out = self.dropout(self.feedforward(norm1_out))
        norm2_out = self.norm2(norm1_out + ff_out)

        return norm2_out

class Model(nn.Module):
    def __init__(self, embed_dim, vocab_size, context_len, n_heads, n_layers, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embedding = TokenEmbedding(self.embed_dim, self.vocab_size)
        self.pe = PositionalEncoding(self.context_len, self.embed_dim)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(self.embed_dim, self.context_len, self.n_heads) for _ in range(self.n_layers)])

        self.norm = nn.LayerNorm(self.embed_dim)

        self.linear = nn.Linear(self.embed_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embeddings = self.embedding(x)
        positional_encoding = self.pe(embeddings)
        
        transformer_blocks = self.transformer_blocks(positional_encoding)
        transformer_out = self.dropout(transformer_blocks)
        
        norm = self.norm(transformer_out)

        out = self.linear(norm)

        return out

def generate_text(model, epoch, tokenizer, device, prompt_tokens, max_len=75, top_k=20, temperature=0.7):
    model.eval()
    tokens = prompt_tokens.copy()
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    special_token_ids = {tokenizer.encode(token, allowed_special={token})[0] for token in tokenizer.special_tokens_set if tokenizer.encode(token, allowed_special={token})}

    with torch.no_grad():
        for _ in range(max_len):
            input_tokens = tokens[:, -model.context_len:]
            logits = model(input_tokens)
            logits = logits[:, -1, :] / temperature

            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx.item()].item()

            tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=1)
            
            if next_token in special_token_ids:
                break

    generated_text = tokenizer.decode(tokens[0].tolist())

    with open(rf'model_output\epoch_{epoch}.txt', 'w') as f:
        f.write(generated_text)

    return generated_text

def save_checkpoint(epoch, model, optimizer, scheduler, cosine_scheduler, scaler, val_loss, model_weights_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
    }, os.path.join(model_weights_path, f'checkpoint_epoch_{epoch}.pt'))

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, cosine_scheduler, scaler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    return epoch, val_loss

BATCH_SIZE = 4
CONTEXT_LEN = 512
EMBED_DIM = 256
VOCAB_SIZE = vocab  # 100,277
N_HEADS = 4
N_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 5
ACCUMULATION_STEPS = 8
PATIENCE = 3
WARMUP_STEPS = 500

print(f"Training on {DEVICE}")

train_len = int(len(tokens) * 0.8)
train_tokens = tokens[:train_len]
val_tokens = tokens[train_len:]

train_data = FineWebData(train_tokens, CONTEXT_LEN)
val_data = FineWebData(val_tokens, CONTEXT_LEN)

print(f"Train dataset length: {len(train_data)}")
print(f"Validation dataset length: {len(val_data)}")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = Model(EMBED_DIM, VOCAB_SIZE, CONTEXT_LEN, N_HEADS, N_LAYERS).to(DEVICE)

summary(model, input_size=(BATCH_SIZE, CONTEXT_LEN), dtypes=[torch.long], depth=3)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

def warmup_lambda(step):
    if step < WARMUP_STEPS:
        return float(step) / WARMUP_STEPS
    return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader) - WARMUP_STEPS)
scaler = GradScaler()

def train(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps, step):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(device_type=device):
            output = model(inputs)
            outputs = output.view(-1, output.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            if step < WARMUP_STEPS:
                scheduler.step()
            else:
                cosine_scheduler.step()

        total_loss += loss.item() * inputs.size(0) * accumulation_steps
        total_tokens += inputs.size(0) * inputs.size(1)

        if idx % 100 == 0:
            progress_bar.set_postfix({"Batch Loss": f"{loss.item() * accumulation_steps:.4f}"})

    avg_loss = total_loss / len(train_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Train Total Loss: {total_loss:.4f}, Dataset Size: {len(train_loader.dataset)}")
    print(f"Epoch Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity, step

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        with autocast(device_type=device):
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)

                output = model(inputs)
                output = output.view(-1, output.size(-1))
                targets = targets.view(-1)

                loss = criterion(output, targets)

                total_loss += loss.item() * inputs.size(0)
                total_tokens += inputs.size(0) * inputs.size(1)

                progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    print(f"Val Total Loss: {total_loss:.4f}, Dataset Size: {len(val_loader.dataset)}")
    print(f"Validation Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    return avg_loss, perplexity

prompt_tokens = tokenizer.encode("The quick brown fox")
model_weights_path = r'model_weights'
os.makedirs(model_weights_path, exist_ok=True)

best_val_loss = float('inf')
epochs_no_improve = 0
global_step = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, train_perplexity, global_step = train(model, train_loader, criterion, optimizer, DEVICE, scaler, ACCUMULATION_STEPS, global_step)
    val_loss, val_perplexity = validate(model, val_loader, criterion, DEVICE)
    
    save_checkpoint(epoch, model, optimizer, scheduler, cosine_scheduler, scaler, val_loss, model_weights_path)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(model_weights_path, f'best_model_epoch_{epoch}.pt'))
        print(f"Saved best model with validation loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    print("\nGenerated Text:")
    generated_text = generate_text(model, epoch, tokenizer, DEVICE, prompt_tokens, max_len=75, top_k=20, temperature=0.7)
    print(generated_text)