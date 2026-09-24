import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Load the text ---
with open("torah.txt", encoding="utf-8") as f:
    text = f.read()

# --- Vocabulary: give every distinct character an integer ---
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}    # string to integer
itos = {i: ch for i, ch in enumerate(chars)}    # integer to string

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

print("Vocabulary size:", vocab_size)
print("Round trip OK:", decode(encode(text)) == text)

# --- Train/validation split: hold out every 10th block of 100 verses ---
verses = text.splitlines()
train_text = "".join(v + "\n" for i, v in enumerate(verses) if (i // 100) % 10 != 0)
val_text = "".join(v + "\n" for i, v in enumerate(verses) if (i // 100) % 10 == 0)
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)
print(f"Train: {len(train_data):,} characters | Validation: {len(val_data):,} characters")

# --- Baselines: simple counting models, scored on validation data ---
print(f"Uniform: {math.log2(vocab_size):.3f} bits/char")

counts = torch.bincount(train_data, minlength=vocab_size).float() + 1
p = counts / counts.sum()
print(f"Unigram: {-torch.log2(p[val_data]).mean().item():.3f} bits/char")

pair_ids = train_data[:-1] * vocab_size + train_data[1:]
pair_counts = torch.bincount(pair_ids, minlength=vocab_size**2).float() + 1
P = pair_counts.reshape(vocab_size, vocab_size)
P = P / P.sum(dim=1, keepdim=True)
print(f"Bigram: {-torch.log2(P[val_data[:-1], val_data[1:]]).mean().item():.3f} bits/char")

# --- Batches ---
torch.manual_seed(1337)    # same random choices every run, so results are reproducible

BLOCK_SIZE = 64    # characters per training sequence
BATCH_SIZE = 32    # sequences per batch

def get_batch(data):
    starts = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    offsets = torch.arange(BLOCK_SIZE)
    x = data[starts.unsqueeze(1) + offsets]        # inputs:  (BATCH_SIZE, BLOCK_SIZE)
    y = data[starts.unsqueeze(1) + offsets + 1]    # targets: the same windows shifted one character on
    return x, y

# --- The model ---
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # row i of this table holds the scores for which character comes after character i
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.table(x)    # (BATCH_SIZE, BLOCK_SIZE, vocab_size)

def loss_fn(logits, y):
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

def estimate_loss(model, data, n_batches=50):
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data)
            total += loss_fn(model(x), y).item()
    return total / n_batches / math.log(2)    # average, converted from nats to bits

# --- Training ---
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for step in range(3001):
    x, y = get_batch(train_data)
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step:5d} | train {estimate_loss(model, train_data):.3f} | val {estimate_loss(model, val_data):.3f} bits/char")