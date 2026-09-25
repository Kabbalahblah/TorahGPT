import math
import sys
import time
import torch
import torch.nn.functional as F
from model import GPT

# --- Settings ---
DATASET = sys.argv[1] if len(sys.argv) > 1 else "torah"    # run as: python3 train.py torah
BLOCK_SIZE = 128        # context length: how many previous characters the model sees
BATCH_SIZE = 32         # sequences per training step
N_EMBD = 128            # length of the vector representing each position
N_HEAD = 4              # attention heads per block
N_LAYER = 4             # number of transformer blocks
DROPOUT = 0.2           # fraction of values randomly zeroed during training
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 250     # steps between validation checks

torch.manual_seed(1337)

# --- Data ---
with open(f"{DATASET}.txt", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s]

# hold out every 10th block of 100 verses (the text has one verse per line)
verses = text.splitlines()
train_text = "".join(v + "\n" for i, v in enumerate(verses) if (i // 100) % 10 != 0)
val_text = "".join(v + "\n" for i, v in enumerate(verses) if (i // 100) % 10 == 0)
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)
print(f"{DATASET}: vocabulary {vocab_size} | train {len(train_data):,} | val {len(val_data):,} characters")

def get_batch(data):
    starts = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    offsets = torch.arange(BLOCK_SIZE)
    x = data[starts.unsqueeze(1) + offsets]
    y = data[starts.unsqueeze(1) + offsets + 1]
    return x, y

def loss_fn(logits, y):
    B, T, V = logits.shape
    return F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

def estimate_loss(model, data, n_batches=50):
    model.eval()        # evaluation mode: dropout switched off
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data)
            total += loss_fn(model(x), y).item()
    model.train()       # back to training mode
    return total / n_batches / math.log(2)

# --- Model ---
config = dict(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD,
              n_head=N_HEAD, n_layer=N_LAYER, dropout=DROPOUT)
model = GPT(**config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training ---
best_val = float("inf")
start = time.time()
for step in range(MAX_STEPS + 1):
    if step % EVAL_INTERVAL == 0:
        train_loss = estimate_loss(model, train_data)
        val_loss = estimate_loss(model, val_data)
        note = ""
        if val_loss < best_val:     # keep the model only when it improves on unseen text
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": config, "chars": chars}, f"{DATASET}gpt.pt")
            note = "  (saved)"
        print(f"step {step:5d} | train {train_loss:.3f} | val {val_loss:.3f} bits/char | {time.time() - start:.0f}s{note}")
    x, y = get_batch(train_data)
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Best validation loss: {best_val:.3f} bits/char, saved in {DATASET}gpt.pt")