import torch
import torchaudio
import json
import os
import uuid
from einops import rearrange
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Define hyperparameters for sweeping
Epochs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
CFG = [0, 1, 2, 3]
Prompts = ["techno", "tech house", "rock and roll", "r&b", "super dead drums", 
           "trap", "drums that smack", "country", "r&b", "blues", "soul"]
Steps = [50, 100, 250]

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")

# Model configuration file path
model_config_file = "./model_config.json"

# Load model configuration
with open(model_config_file, "r") as f:
    model_config = json.load(f)

sample_rate = model_config["sample_rate"]
duration_seconds = 47  # Set duration to 47 seconds
sample_size = int(sample_rate * duration_seconds)

# Directory to save audio outputs
output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

def load_model(epoch):
    """Loads the model checkpoint corresponding to the given epoch."""
    model_ckpt_file = f"epoch={epoch}.ckpt"

    print(f"Loading model: {model_ckpt_file}")
    checkpoint = torch.load(model_ckpt_file, map_location=device)
    model_state_dict = checkpoint["state_dict"]

    model = create_model_from_config(model_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model

def generate_audio(model, prompt, steps, cfg_scale):
    """Generates audio using the diffusion model."""
    full_prompt = f"Generate Drum Kit: {prompt}"  # Add prefix

    conditioning = [[{
        "prompt": full_prompt,
        "seconds_start": 0,
        "seconds_total": duration_seconds  # Set duration to 47 seconds
    }]]

    print(f"Generating: {full_prompt} | Steps: {steps} | CFG: {cfg_scale}")
    
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Reshape the output
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    return output

def save_audio(output, epoch, cfg, prompt, steps):
    """Saves the generated audio as a WAV file."""
    filename = f"{output_dir}/epoch={epoch}_cfg={cfg}_prompt={prompt.replace(' ', '_')}_steps={steps}.wav"
    
    print(f"Saving audio: {filename}")
    torchaudio.save(filename, output, sample_rate)

# Loop through all parameter combinations
for epoch in Epochs:
    model = load_model(epoch)

    for cfg in CFG:
        for prompt in Prompts:
            for steps in Steps:
                output = generate_audio(model, prompt, steps, cfg)
                save_audio(output, epoch, cfg, prompt, steps)

print("Inference completed. Check the 'generated_audio' folder for results.")
