import argparse
import os

# Run rename_state_dict.py to get the model in the correct format

parser = argparse.ArgumentParser()
parser.add_argument("--model-s3-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--model-config", type=str, required=True)
args = parser.parse_args()

tmp_path = f"{args.output_path}/{args.model_path}.ckpt"
file_name = os.path.basename(args.model_path)

os.system(f"aws s3 cp {args.model_s3_path} {tmp_path}")
os.system(f"python rename_state_dict.py --model-path {tmp_path} --output-path {tmp_path}")
os.system(f"python unwrap_model.py --model-config {args.model_config} --ckpt-path {tmp_path} --name {args.output_path}/{file_name}")

os.remove(tmp_path)
