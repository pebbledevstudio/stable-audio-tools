# Docker Setup for Stable Audio Tools

This guide explains how to use Docker to run the stable-audio-tools training with command line argument overrides.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t stable-audio-tools .
```

### 2. Run with Default Configuration

```bash
docker run --gpus all -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/wandb:/app/wandb stable-audio-tools
```

### 3. Run with Command Line Overrides

The training script uses prefigure which automatically supports command line argument overrides. Any arguments you pass will override the corresponding values in `defaults.ini`:

```bash
# Override batch size and number of GPUs
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py --batch-size 4 --num-gpus 2

# Override model and dataset configs
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py --model-config custom_model.json --dataset-config custom_dataset.json

# Override training parameters
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py --checkpoint-every 5000 --accum-batches 2 --precision "32"
```

## Using Docker Compose

### 1. Run with Default Settings

```bash
docker-compose up
```

### 2. Run with Custom Arguments

```bash
docker-compose run stable-audio-training python train.py --batch-size 8 --num-gpus 1
```

### 3. Interactive Mode

```bash
docker-compose run stable-audio-training bash
# Inside the container:
python train.py --help  # See all available options
python train.py --batch-size 2 --checkpoint-every 1000
```

## Available Command Line Arguments

The following arguments can be used to override `defaults.ini` values:

- `--name`: Name of the run
- `--batch-size`: Batch size for training
- `--num-gpus`: Number of GPUs to use
- `--num-nodes`: Number of nodes for distributed training
- `--strategy`: Multi-GPU strategy (e.g., "deepspeed")
- `--precision`: Training precision (e.g., "16-mixed", "32")
- `--num-workers`: Number of CPU workers for DataLoader
- `--seed`: Random seed
- `--accum-batches`: Gradient accumulation batches
- `--checkpoint-every`: Steps between checkpoints
- `--ckpt-path`: Checkpoint path to resume from
- `--pretrained-ckpt-path`: Pre-trained model checkpoint
- `--pretransform-ckpt-path`: Pre-transform checkpoint
- `--model-config`: Model configuration JSON file
- `--dataset-config`: Dataset configuration JSON file
- `--save-dir`: Directory to save checkpoints
- `--gradient-clip-val`: Gradient clipping value
- `--remove-pretransform-weight-norm`: Remove weight norm option
- `--s3-checkpoint-bucket`: S3 bucket for checkpoints
- `--s3-checkpoint-prefix`: S3 prefix for checkpoints
- `--s3-demo-bucket`: S3 bucket for demo uploads
- `--s3-demo-prefix`: S3 prefix for demo uploads
- `--s3-checkpoint-mode`: Checkpoint upload mode
- `--s3-delete-after-upload`: Delete local files after S3 upload

## Volume Mounts

The Docker setup includes the following volume mounts:

- `./checkpoints:/app/checkpoints` - Preserves training checkpoints
- `./wandb:/app/wandb` - Preserves Weights & Biases logs
- `./dataset_flat_clean:/app/dataset_flat_clean` - Mounts local dataset

## GPU Support

The Docker setup is configured for NVIDIA GPU support. Make sure you have:

1. NVIDIA Docker runtime installed
2. Use `--gpus all` flag when running with `docker run`
3. Docker Compose automatically configures GPU access

## Environment

The Docker container:

1. **Automatically activates the virtual environment** before running any commands
2. Installs all dependencies from `requirements_p39.txt`
3. Includes system dependencies like `ffmpeg` for audio processing
4. Uses Python 3.9 as the base environment

## Examples

### Training with Custom Batch Size and Checkpointing

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py \
    --batch-size 16 \
    --checkpoint-every 2000 \
    --name "my-custom-training" \
    --precision "16-mixed"
```

### Distributed Training

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py \
    --num-gpus 4 \
    --strategy "deepspeed" \
    --batch-size 8
```

### Resume from Checkpoint

```bash
docker run --gpus all \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/wandb:/app/wandb \
  stable-audio-tools \
  python train.py \
    --ckpt-path "/app/checkpoints/path/to/checkpoint.ckpt" \
    --batch-size 4
```

## Troubleshooting

### Check Available Arguments

```bash
docker run stable-audio-tools python train.py --help
```

### Interactive Debugging

```bash
docker run -it --gpus all stable-audio-tools bash
# Inside container:
source venv/bin/activate  # Already activated by entrypoint
python train.py --help
```

### View Current Configuration

The training script will show the final configuration (after overrides) in the logs when it starts.


