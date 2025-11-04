import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
args = parser.parse_args()

model = torch.load(args.model_path, map_location='cpu')

model["state_dict"] = model["module"]
#model["state_dict"]["diffusion_ema.initted"] = model["state_dict"]["diffusion_ema.initted"].unsqueeze(0)
#model["state_dict"]["diffusion_ema.step"] = model["state_dict"]["diffusion_ema.step"].unsqueeze(0)

torch.save(model, args.output_path)
