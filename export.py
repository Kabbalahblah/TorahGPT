import json
import os
import sys
import torch
from model import GPT

DATASET = sys.argv[1] if len(sys.argv) > 1 else "torah"    # run as: python3 export.py torah

checkpoint = torch.load(f"{DATASET}gpt.pt")
config = checkpoint["config"]
model = GPT(**config)
model.load_state_dict(checkpoint["model"])
model.eval()

os.makedirs("docs/models", exist_ok=True)
example = torch.zeros((1, 16), dtype=torch.long)       # any valid input, used to trace the model
length = torch.export.Dim("length", min=1, max=config["block_size"])
torch.onnx.export(
    model, (example,), f"docs/models/{DATASET}.onnx",
    input_names=["ids"], output_names=["logits"],
    dynamic_shapes={"idx": {1: length}},               # the context length can vary
    external_data=False,                                # keep weights inside the one file
    dynamo=True,
)

# the vocabulary and context length, which the web page needs to encode and decode text
with open(f"docs/models/{DATASET}.json", "w", encoding="utf-8") as f:
    json.dump({"chars": checkpoint["chars"], "block_size": config["block_size"]}, f, ensure_ascii=False)

print(f"Exported docs/models/{DATASET}.onnx and docs/models/{DATASET}.json")