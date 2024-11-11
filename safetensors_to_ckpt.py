import torch
from safetensors import safe_open
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Convert safetensors file to checkpoint")
parser.add_argument("safetensors_path", type=str, help="Path to the input safetensors file")
parser.add_argument("checkpoint_path", type=str, help="Path to the output checkpoint file")
args = parser.parse_args()

# Load the model weights from safetensors
model_dict = {}
with safe_open(args.safetensors_path, framework="pt") as f:
    for key in f.keys():
        model_dict[key] = f.get_tensor(key)

# Save the model weights as a checkpoint
torch.save(model_dict, args.checkpoint_path)
print(f"Conversion complete: {args.checkpoint_path}")
