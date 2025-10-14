from prefigure.prefigure import get_all_args, push_wandb_config
import json
import os
import torch
import pytorch_lightning as pl
import random

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.utils import copy_state_dict

# --- helpers ---
from pydub import AudioSegment
import numpy as np
import io
import tempfile
import wandb
import soundfile as sf
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import boto3
from datetime import datetime

def audio_to_mp3_bytes(audio: np.ndarray, sample_rate: int, vbr_quality: int = 2) -> bytes:
    # audio: float32 [-1,1], shape (C,T)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).T  # (T,C)
    buf_wav = io.BytesIO()
    sf.write(buf_wav, pcm, samplerate=sample_rate, subtype="PCM_16", format="WAV")
    buf_wav.seek(0)
    seg = AudioSegment.from_file(buf_wav, format="wav")
    buf_mp3 = io.BytesIO()
    seg.export(buf_mp3, format="mp3", parameters=["-q:a","2"])
    buf_mp3.seek(0)
    return buf_mp3.read()

def upload_to_s3(data, bucket, key, content_type=None):
    """Upload data to S3"""
    try:
        s3_client = boto3.client('s3')
        if isinstance(data, bytes):
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type or 'application/octet-stream'
            )
        else:
            # For file paths
            s3_client.upload_file(data, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return None

def upload_directory_to_s3(local_dir, bucket, s3_prefix):
    """Upload entire directory to S3, preserving folder structure"""
    try:
        s3_client = boto3.client('s3')
        uploaded_files = []
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')  # Handle Windows paths
                
                print(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
                s3_client.upload_file(local_path, bucket, s3_key)
                uploaded_files.append(f"s3://{bucket}/{s3_key}")
        
        return uploaded_files
    except Exception as e:
        print(f"Failed to upload directory to S3: {e}")
        return None

# --- demo callback ---
class WandBMP3DemoCallback(pl.Callback):
    def __init__(self, sample_rate, demo_fn, limit=4, s3_bucket=None, s3_prefix=None):
        self.sample_rate = sample_rate
        self.demo_fn = demo_fn  # function (pl_module, trainer) -> [(name, np.ndarray)]
        self.limit = limit
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Debug: Show when callback is triggered (but not necessarily when demo runs)
        if trainer.global_step % 100 == 0:  # Only log every 100 steps to avoid spam
            print(f"[DEBUG] WandBMP3DemoCallback.on_train_batch_end at step {trainer.global_step}")
        
        print(f"[DEBUG] WandBMP3DemoCallback calling demo_fn at step {trainer.global_step}")
        pairs = self.demo_fn(pl_module, trainer)
        print(f"[DEBUG] WandBMP3DemoCallback received {len(pairs)} pairs from demo_fn")
        
        if len(pairs) == 0:
            print(f"[DEBUG] No demo pairs returned, skipping demo generation at step {trainer.global_step}")
            return
        
        for i,(name,audio) in enumerate(pairs[:self.limit]):
            print(f"[DEBUG] Processing demo {i+1}/{len(pairs[:self.limit])}: {name}, audio shape: {audio.shape}")
            
            try:
                mp3_bytes = audio_to_mp3_bytes(audio, self.sample_rate)
                print(f"[DEBUG] Converted to MP3: {len(mp3_bytes)} bytes")
                
                # Create timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{trainer.global_step:08d}_{timestamp}.mp3"
                print(f"[DEBUG] Generated filename: {filename}")
                
                # Log to WandB
                print(f"[DEBUG] Logging to WandB...")
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(mp3_bytes); tmp.flush()
                    wandb.log({"demo_audio": wandb.Audio(tmp.name,
                               sample_rate=self.sample_rate,
                               caption=name)})
                    os.remove(tmp.name)
                print(f"[DEBUG] Successfully logged to WandB")
                
                # Upload to S3 if configured
                if self.s3_bucket and self.s3_prefix:
                    print(f"[DEBUG] Uploading to S3: bucket={self.s3_bucket}, prefix={self.s3_prefix}")
                    s3_key = f"{self.s3_prefix}/{filename}"
                    s3_path = upload_to_s3(mp3_bytes, self.s3_bucket, s3_key, content_type="audio/mpeg")
                    if s3_path:
                        print(f"Demo audio uploaded to S3: {s3_path}")
                        wandb.log({"demo_s3_path": s3_path})
                    else:
                        print(f"[DEBUG] S3 upload failed for {filename}")
                else:
                    print(f"[DEBUG] S3 not configured, skipping S3 upload")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process demo {name}: {e}")
                import traceback
                traceback.print_exc()

# --- checkpoint uploader ---
class S3CheckpointCallback(pl.Callback):
    def __init__(self, s3_bucket=None, s3_prefix=None, upload_mode='single_file', delete_after_upload=True):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.upload_mode = upload_mode  # 'single_file' or 'full_directory'
        self.delete_after_upload = delete_after_upload
        self.last_uploaded_step = -1

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only run on checkpoint steps
        if trainer.global_step % trainer.checkpoint_callback._every_n_train_steps != 0:
            return
            
        # Skip if just started (step 0) or if we've already processed this step
        if trainer.global_step == 0 or self.last_uploaded_step == trainer.global_step:
            return
            
        print(f"S3CheckpointCallback checking for checkpoint at step {trainer.global_step}")
        
        if not self.s3_bucket or not self.s3_prefix:
            print("S3 checkpoint upload not configured - skipping")
            return
        
        # Wait for checkpoint to be saved (it should exist by now since we're in batch_end)
        import time
        max_wait = 10  # Maximum 10 seconds
        wait_time = 0
        ckpt_path = None
        
        while wait_time < max_wait:
            # Try to get the checkpoint path
            potential_paths = [
                trainer.checkpoint_callback.last_model_path,
                trainer.checkpoint_callback.best_model_path
            ]
            
            for path in potential_paths:
                if path and os.path.exists(path):
                    # Handle both files and directories
                    try:
                        if os.path.isdir(path):
                            # For directories, check if they contain files
                            files_in_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                            if files_in_dir:  # Directory has files
                                ckpt_path = path
                                break
                        else:
                            # For files, check if file size is stable (not still being written)
                            size1 = os.path.getsize(path)
                            time.sleep(0.1)
                            size2 = os.path.getsize(path)
                            if size1 == size2 and size1 > 0:  # File size is stable and non-zero
                                ckpt_path = path
                                break
                    except OSError:
                        continue
            
            if ckpt_path:
                break
                
            time.sleep(0.5)
            wait_time += 0.5
        
        print(f"Checkpoint path: {ckpt_path}")
        print(f"Path exists: {os.path.exists(ckpt_path) if ckpt_path else False}")
        if ckpt_path and os.path.exists(ckpt_path):
            if os.path.isdir(ckpt_path):
                files_in_dir = [f for f in os.listdir(ckpt_path) if os.path.isfile(os.path.join(ckpt_path, f))]
                print(f"Directory contains {len(files_in_dir)} files")
            else:
                print(f"File size: {os.path.getsize(ckpt_path)} bytes")
        
        if ckpt_path and os.path.exists(ckpt_path):
            # Create S3 key with timestamp and step
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_filename = os.path.basename(ckpt_path)
            s3_prefix_with_timestamp = f"{self.s3_prefix}/{timestamp}_step_{trainer.global_step:08d}_{ckpt_filename}"
            print(f"S3 bucket: {self.s3_bucket}")
            print(f"S3 prefix: {s3_prefix_with_timestamp}")
            print(f"Upload mode: {self.upload_mode}")
            
            if os.path.isdir(ckpt_path):
                if self.upload_mode == 'full_directory':
                    # Upload entire directory
                    print(f"Checkpoint is a directory, uploading all files...")
                    uploaded_files = upload_directory_to_s3(ckpt_path, self.s3_bucket, s3_prefix_with_timestamp)
                    if uploaded_files:
                        print(f"Checkpoint directory uploaded to S3: {len(uploaded_files)} files")
                        # Log the S3 prefix where files were uploaded
                        s3_path = f"s3://{self.s3_bucket}/{s3_prefix_with_timestamp}/"
                        wandb.log({
                            "checkpoint_s3_path": s3_path, 
                            "checkpoint_step": trainer.global_step,
                            "checkpoint_files_count": len(uploaded_files)
                        })
                        self.last_uploaded_step = trainer.global_step
                        
                        # Delete the entire checkpoint directory to save storage
                        if self.delete_after_upload:
                            try:
                                import shutil
                                shutil.rmtree(ckpt_path)
                                print(f"Deleted local checkpoint directory: {ckpt_path}")
                            except OSError as e:
                                print(f"Failed to delete local directory {ckpt_path}: {e}")
                        else:
                            print("Local checkpoint directory preserved (delete_after_upload=False)")
                    else:
                        print("S3 directory upload failed")
                else:
                    # Look for specific model states file
                    model_states_file = os.path.join(ckpt_path, "checkpoint", "mp_rank_00_model_states.pt")
                    if os.path.exists(model_states_file):
                        print(f"Found model states file: {model_states_file}")
                        s3_key = f"{s3_prefix_with_timestamp}/mp_rank_00_model_states.pt"
                        s3_path = upload_to_s3(model_states_file, self.s3_bucket, s3_key)
                        print(f"S3 path returned: {s3_path}")
                        
                        if s3_path:
                            print(f"Model states file uploaded to S3: {s3_path}")
                            wandb.log({"checkpoint_s3_path": s3_path, "checkpoint_step": trainer.global_step})
                            self.last_uploaded_step = trainer.global_step
                            
                            # Delete the uploaded file to save storage
                            if self.delete_after_upload:
                                try:
                                    os.remove(model_states_file)
                                    print(f"Deleted local model states file: {model_states_file}")
                                except OSError as e:
                                    print(f"Failed to delete local file {model_states_file}: {e}")
                            else:
                                print("Local model states file preserved (delete_after_upload=False)")
                        else:
                            print("S3 upload failed - s3_path is None")
                    else:
                        print(f"Model states file not found at: {model_states_file}")
                        print("Available files in checkpoint directory:")
                        for root, dirs, files in os.walk(ckpt_path):
                            for file in files:
                                rel_path = os.path.relpath(os.path.join(root, file), ckpt_path)
                                print(f"  {rel_path}")
            else:
                # Upload single file
                print(f"Checkpoint is a file, uploading...")
                s3_key = f"{s3_prefix_with_timestamp}/{ckpt_filename}"
                s3_path = upload_to_s3(ckpt_path, self.s3_bucket, s3_key)
                print(f"S3 path returned: {s3_path}")
                
                if s3_path:
                    print(f"Checkpoint uploaded to S3: {s3_path}")
                    wandb.log({"checkpoint_s3_path": s3_path, "checkpoint_step": trainer.global_step})
                    self.last_uploaded_step = trainer.global_step
                    
                    # Delete the uploaded file to save storage
                    if self.delete_after_upload:
                        try:
                            os.remove(ckpt_path)
                            print(f"Deleted local checkpoint file: {ckpt_path}")
                        except OSError as e:
                            print(f"Failed to delete local file {ckpt_path}: {e}")
                    else:
                        print("Local checkpoint file preserved (delete_after_upload=False)")
                else:
                    print("S3 upload failed - s3_path is None")
        else:
            print(f"Checkpoint upload skipped - file doesn't exist after waiting {max_wait} seconds")

def create_demo_function_from_config(model_config, **kwargs):
    """Create a demo function that extracts audio generation logic from model config"""
    import gc
    import torch
    from einops import rearrange
    
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    demo_config = training_config.get("demo", {})
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    demo_every = demo_config.get("demo_every", 2000)
    
    print(f"[DEBUG] Creating demo function for model_type='{model_type}', demo_every={demo_every}")
    print(f"[DEBUG] Demo config: {demo_config}")
    print(f"[DEBUG] Available kwargs: {list(kwargs.keys())}")
    
    if model_type == 'autoencoder':
        # Use closure for last_demo_step tracking
        last_demo_step = [-1]  # Use list to make it mutable in closure
        
        def demo_fn(pl_module, trainer):
            print(f"[DEBUG] Autoencoder demo_fn called at step {trainer.global_step}")
            print(f"[DEBUG] Checking timing: (step-1) % demo_every = ({trainer.global_step-1}) % {demo_every} = {(trainer.global_step - 1) % demo_every}")
            print(f"[DEBUG] Last demo step: {last_demo_step[0]}")
            
            # Use the same timing logic as original callbacks
            if (trainer.global_step - 1) % demo_every != 0 or last_demo_step[0] == trainer.global_step:
                print(f"[DEBUG] Skipping demo - timing check failed or already processed this step")
                return []
            
            last_demo_step[0] = trainer.global_step
            print(f"[DEBUG] Generating autoencoder demo at step {trainer.global_step}")
            
            # Get demo data from training dataloader
            demo_dl = kwargs.get('demo_dl')
            print(f"[DEBUG] Demo dataloader available: {demo_dl is not None}")
            if demo_dl is None:
                print(f"[DEBUG] No demo dataloader provided, returning empty list")
                return []
                
            try:
                demo_reals, _ = next(iter(demo_dl))
                demo_reals = demo_reals[:4].to(pl_module.device)
                
                with torch.no_grad():
                    if hasattr(pl_module, 'model_ema') and pl_module.model_ema is not None:
                        model = pl_module.model_ema.ema_model
                    else:
                        model = pl_module.model
                    
                    # Encode and decode
                    if hasattr(model, 'encode') and hasattr(model, 'decode'):
                        encoded = model.encode(demo_reals)
                        fakes = model.decode(encoded)
                    else:
                        fakes = model(demo_reals)
                
                # Interleave reals and fakes
                reals_fakes = rearrange([demo_reals, fakes], 'i b d n -> (b i) d n')
                reals_fakes = rearrange(reals_fakes, 'b d n -> d (b n)')
                
                # Convert to numpy float32 in range [-1, 1]
                audio_np = reals_fakes.cpu().numpy().astype(np.float32)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                
                return [("autoencoder_demo", audio_np)]
                
            except Exception as e:
                print(f"Demo generation failed: {e}")
                return []
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                pl_module.train()
        
        return demo_fn
    
    elif model_type == 'diffusion_uncond':
        # Use closure for last_demo_step tracking
        last_demo_step = [-1]  # Use list to make it mutable in closure
        
        def demo_fn(pl_module, trainer):
            print(f"[DEBUG] Diffusion uncond demo_fn called at step {trainer.global_step}")
            print(f"[DEBUG] Checking timing: (step-1) % demo_every = ({trainer.global_step-1}) % {demo_every} = {(trainer.global_step - 1) % demo_every}")
            print(f"[DEBUG] Last demo step: {last_demo_step[0]}")
            
            # Use the same timing logic as original callbacks
            if (trainer.global_step - 1) % demo_every != 0 or last_demo_step[0] == trainer.global_step:
                print(f"[DEBUG] Skipping diffusion demo - timing check failed or already processed this step")
                return []
            
            last_demo_step[0] = trainer.global_step
            print(f"[DEBUG] Generating diffusion uncond demo at step {trainer.global_step}")
                
            try:
                demo_steps = demo_config.get("demo_steps", 250)
                num_demos = demo_config.get("num_demos", 4)
                demo_samples = sample_size // pl_module.diffusion.downsampling_ratio if hasattr(pl_module, 'diffusion') else sample_size
                
                noise = torch.randn([num_demos, pl_module.diffusion.io_channels, demo_samples]).to(pl_module.device)
                
                with torch.cuda.amp.autocast():
                    from stable_audio_tools.inference.generation import sample
                    fakes = sample(pl_module.diffusion_ema if hasattr(pl_module, 'diffusion_ema') else pl_module.diffusion, 
                                 noise, demo_steps, 0)

                    if pl_module.diffusion.pretransform is not None:
                        fakes = pl_module.diffusion.pretransform.decode(fakes)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')
                
                # Convert to numpy float32 in range [-1, 1]
                audio_np = fakes.cpu().numpy().astype(np.float32)
                audio_np = np.clip(audio_np / audio_np.max(), -1.0, 1.0)
                
                return [("diffusion_uncond_demo", audio_np)]
                
            except Exception as e:
                print(f"Demo generation failed: {e}")
                return []
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                pl_module.train()
        
        return demo_fn
    
    elif model_type == 'diffusion_cond':
        # Use closure for last_demo_step tracking
        last_demo_step = [-1]  # Use list to make it mutable in closure
        
        def demo_fn(pl_module, trainer):
            print(f"[DEBUG] Diffusion cond demo_fn called at step {trainer.global_step}")
            print(f"[DEBUG] Checking timing: (step-1) % demo_every = ({trainer.global_step-1}) % {demo_every} = {(trainer.global_step - 1) % demo_every}")
            print(f"[DEBUG] Last demo step: {last_demo_step[0]}")
            
            # Use the same timing logic as original callbacks
            if (trainer.global_step - 1) % demo_every != 0 or last_demo_step[0] == trainer.global_step:
                print(f"[DEBUG] Skipping diffusion cond demo - timing check failed or already processed this step")
                return []
            
            last_demo_step[0] = trainer.global_step
            print(f"[DEBUG] Generating diffusion cond demo at step {trainer.global_step}")
                
            try:
                demo_steps = demo_config.get("demo_steps", 250)
                num_demos = demo_config.get("num_demos", 4)
                demo_cfg_scales = demo_config.get("demo_cfg_scales", [3, 5, 7])
                demo_conditioning = demo_config.get("demo_cond", [])
                
                print(f"[DEBUG] Demo parameters: steps={demo_steps}, num_demos={num_demos}, cfg_scales={demo_cfg_scales}")
                print(f"[DEBUG] Demo conditioning prompts: {len(demo_conditioning)}")
                
                if not demo_conditioning:
                    print(f"[DEBUG] No conditioning prompts available, skipping demo")
                    return []
                
                # Calculate demo samples based on pretransform downsampling
                if hasattr(pl_module, 'diffusion') and pl_module.diffusion.pretransform is not None:
                    demo_samples = sample_size // pl_module.diffusion.pretransform.downsampling_ratio
                else:
                    demo_samples = sample_size
                
                # Generate demos for each CFG scale
                results = []
                for cfg_scale in demo_cfg_scales[:1]:  # Just use first CFG scale for simplicity
                    print(f"[DEBUG] Generating demo for CFG scale {cfg_scale}")
                    
                    # Use the first conditioning prompt
                    cond = demo_conditioning[0] if demo_conditioning else {}
                    
                    noise = torch.randn([num_demos, pl_module.diffusion.io_channels, demo_samples]).to(pl_module.device)
                    
                    with torch.cuda.amp.autocast():
                        # Import the sampling function
                        from stable_audio_tools.inference.generation import sample
                        
                        model = pl_module.diffusion_ema.model if hasattr(pl_module, 'diffusion_ema') and pl_module.diffusion_ema is not None else pl_module.diffusion.model
                        
                        # Prepare conditioning inputs (simplified)
                        cond_inputs = {}
                        
                        # Use sample function for all cases
                        fakes = sample(model, noise, demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                        
                        if pl_module.diffusion.pretransform is not None:
                            fakes = pl_module.diffusion.pretransform.decode(fakes)

                    # Put the demos together
                    fakes = rearrange(fakes, 'b d n -> d (b n)')
                    
                    # Convert to numpy float32 in range [-1, 1]
                    audio_np = fakes.cpu().numpy().astype(np.float32)
                    audio_np = np.clip(audio_np / audio_np.max(), -1.0, 1.0)
                    
                    results.append((f"diffusion_cond_cfg_{cfg_scale}", audio_np))
                
                return results
                
            except Exception as e:
                print(f"[ERROR] Diffusion cond demo generation failed: {e}")
                import traceback
                traceback.print_exc()
                return []
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                pl_module.train()
        
        return demo_fn
    
    else:
        # For other model types, return a dummy function that does nothing
        print(f"[DEBUG] No demo function available for model_type='{model_type}', creating dummy function")
        def demo_fn(pl_module, trainer):
            print(f"[DEBUG] Dummy demo function called for unsupported model type '{model_type}'")
            return []
        return demo_fn

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():

    args = get_all_args()

    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    random.seed(seed)
    torch.manual_seed(seed)

    #Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
    )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))
    
    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    
    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    exc_callback = ExceptionCallback()
    
    if args.save_dir and isinstance(wandb_logger.experiment.id, str):
        checkpoint_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id, "checkpoints") 
    else:
        checkpoint_dir = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    # Create demo function and MP3 demo callback with S3 configuration
    print(f"[DEBUG] Creating demo function...")
    demo_fn = create_demo_function_from_config(model_config, demo_dl=train_dl)
    print(f"[DEBUG] Demo function created successfully")
    
    s3_demo_bucket = getattr(args, 's3_demo_bucket', None)
    s3_demo_prefix = getattr(args, 's3_demo_prefix', None)
    print(f"[DEBUG] Creating WandBMP3DemoCallback with:")
    print(f"[DEBUG]   sample_rate: {model_config['sample_rate']}")
    print(f"[DEBUG]   s3_bucket: {s3_demo_bucket}")
    print(f"[DEBUG]   s3_prefix: {s3_demo_prefix}")
    
    demo_callback = WandBMP3DemoCallback(
        sample_rate=model_config["sample_rate"], 
        demo_fn=demo_fn,
        s3_bucket=s3_demo_bucket,
        s3_prefix=s3_demo_prefix
    )
    print(f"[DEBUG] WandBMP3DemoCallback created successfully")
    
    # Add S3 checkpoint uploader callback
    checkpoint_uploader_callback = S3CheckpointCallback(
        s3_bucket=getattr(args, 's3_checkpoint_bucket', None),
        s3_prefix=getattr(args, 's3_checkpoint_prefix', None),
        upload_mode=getattr(args, 's3_checkpoint_mode', 'single_file'),
        delete_after_upload=getattr(args, 's3_delete_after_upload', True)
    )

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    #Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2, 
                                        contiguous_gradients=True, 
                                        overlap_comm=True, 
                                        reduce_scatter=True, 
                                        reduce_bucket_size=5e8, 
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True
                                        )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto" 

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[ckpt_callback, demo_callback, exc_callback, save_model_config_callback, checkpoint_uploader_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()