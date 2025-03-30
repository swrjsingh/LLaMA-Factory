#!/usr/bin/env python
import os
import sys
from custom_trainer import DetailedSampleTrainer

# Add the current directory to the path so we can import our custom trainer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# This function will monkey patch LLaMA-Factory to use our custom trainer
def patch_llamafactory():
    # Import needed modules after adding to path
    from llamafactory.train.sft import workflow
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    
    # Store the original trainer class
    original_trainer = CustomSeq2SeqTrainer
    
    # Store the original run_sft function
    original_run_sft = workflow.run_sft
    
    # Define our patched run_sft function
    def patched_run_sft(*args, **kwargs):
        # Replace the trainer class in the workflow module
        workflow.CustomSeq2SeqTrainer = DetailedSampleTrainer
        
        try:
            # Run the original function
            result = original_run_sft(*args, **kwargs)
            return result
        finally:
            # Restore the original trainer class
            workflow.CustomSeq2SeqTrainer = original_trainer
    
    # Replace the original function with our patched version
    workflow.run_sft = patched_run_sft
    
    print("✅ LLaMA-Factory has been patched to use the DetailedSampleTrainer")
    return True

# Execute the patch when this script is imported
patch_successful = patch_llamafactory()

if __name__ == "__main__":
    print("This script should be imported, not run directly.")
    print("To use it, add this to your Python path and import it before running LLaMA-Factory.") 