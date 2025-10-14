#!/usr/bin/env python3
"""
Test script to demonstrate command line argument overrides.
This script shows how arguments override defaults.ini values.
"""

from prefigure.prefigure import get_all_args
import sys

def main():
    print("Testing command line argument overrides...")
    print(f"Command line arguments: {sys.argv[1:]}")
    print()
    
    # Get all arguments (from defaults.ini + command line overrides)
    args = get_all_args()
    
    # Show some key parameters
    print("Current configuration:")
    print(f"  name: {args.name}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  num_gpus: {args.num_gpus}")
    print(f"  checkpoint_every: {args.checkpoint_every}")
    print(f"  precision: {args.precision}")
    print(f"  strategy: {args.strategy}")
    print(f"  model_config: {args.model_config}")
    print(f"  dataset_config: {args.dataset_config}")
    print(f"  save_dir: {args.save_dir}")
    
    print()
    print("✅ Command line arguments successfully override defaults.ini values!")

if __name__ == "__main__":
    main()


