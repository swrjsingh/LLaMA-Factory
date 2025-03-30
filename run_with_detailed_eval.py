#!/usr/bin/env python
"""
Run LLaMA-Factory with detailed evaluation saving.
This script patches LLaMA-Factory to save detailed sample-level predictions during evaluation.
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

def ensure_custom_trainer_exists():
    # Check if our custom trainer files exist in the current directory
    custom_trainer_path = Path('custom_trainer.py')
    custom_integration_path = Path('custom_integration.py')
    
    if not custom_trainer_path.exists() or not custom_integration_path.exists():
        print("Error: custom_trainer.py or custom_integration.py not found")
        print("Make sure these files are in the same directory as this script")
        sys.exit(1)
    
    print("✓ Found custom trainer files")

def run_llamafactory_cmd(cmd_args):
    # Import our custom integration to patch LLaMA-Factory
    try:
        print("Patching LLaMA-Factory with DetailedSampleTrainer...")
        import custom_integration
        print("✓ Successfully imported and patched")
    except ImportError as e:
        print(f"Error importing custom integration: {e}")
        sys.exit(1)
    
    # Construct the command
    cmd = ["llamafactory-cli"] + cmd_args
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        # Use subprocess to run the command
        result = subprocess.run(cmd)
        
        # Check return code
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            sys.exit(result.returncode)
        
        print("✓ Command completed successfully")
        
        # Find and report the detailed predictions file
        output_dir = None
        for i, arg in enumerate(cmd_args):
            if arg == "--output_dir" and i+1 < len(cmd_args):
                output_dir = cmd_args[i+1]
                break
        
        if output_dir:
            detailed_preds_path = os.path.join(output_dir, "detailed_predictions.jsonl")
            if os.path.exists(detailed_preds_path):
                print(f"✓ Detailed predictions saved to: {detailed_preds_path}")
            else:
                print(f"! No detailed predictions found at: {detailed_preds_path}")
        
    except Exception as e:
        print(f"Error running command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if our custom files exist
    ensure_custom_trainer_exists()
    
    # Get command-line arguments (skip the script name)
    args = sys.argv[1:]
    
    # Make sure we have arguments
    if not args:
        print("Error: No arguments provided")
        print("Usage: python run_with_detailed_eval.py train --stage sft ...")
        sys.exit(1)
    
    # Add current directory to Python path to find our modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Run the command
    run_llamafactory_cmd(args) 