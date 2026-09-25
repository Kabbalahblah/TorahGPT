import sys
import torch
from model import GPT

DATASET = sys.argv[1] if len(sys.argv) > 1 else "torah"    # run as: python3 sample.py torah
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "\n"         # optional starting text
N_CHARS = 500
TEMPERATURE = 1.0

checkpoint = torch.load(f"{DATASET}gpt.pt")
chars = checkpoint["chars"]
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

model = GPT(**checkpoint["config"])
model.load_state_dict(checkpoint["model"])
model.eval()

prompt = "".join(ch for ch in PROMPT if ch in stoi) or "\n"   # drop characters the model never saw
ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)
text = "".join(itos[i] for i in model.generate(ids, N_CHARS, TEMPERATURE)[0].tolist())

with open("sample.txt", "w", encoding="utf-8") as f:
    f.write(text)
print(text)