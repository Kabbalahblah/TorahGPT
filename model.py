import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Lets each position gather information from itself and earlier positions."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)    # queries, keys and values, computed together
        self.proj = nn.Linear(n_embd, n_embd)       # mixes the heads' outputs back together
        self.dropout = nn.Dropout(dropout)
        # lower-triangular mask: position t may only look at positions 0 to t
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).bool())

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # split each into n_head smaller heads: shape (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # how strongly each position attends to each other position: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))    # no peeking at the future
        att = self.dropout(F.softmax(att, dim=-1))
        out = att @ v                                   # weighted average of the values
        out = out.transpose(1, 2).reshape(B, T, C)      # glue the heads back together
        return self.dropout(self.proj(out))


class Block(nn.Module):
    """One transformer block: attention (communication), then an MLP (computation)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # residual connections: each part adds to x
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)    # what each character is
        self.pos_emb = nn.Embedding(block_size, n_embd)      # where it sits in the window
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)            # vector -> scores for the next character

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.ln_f(x))                       # (B, T, vocab_size)

    def generate(self, ids, n_chars, temperature=1.0):
        with torch.no_grad():
            for _ in range(n_chars):
                logits = self(ids[:, -self.block_size:])[:, -1, :]   # scores for the next character
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
        return ids