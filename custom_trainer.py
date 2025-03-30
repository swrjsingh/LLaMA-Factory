from typing import Dict, List, Optional, Union, Any
import torch
import json
import os
import numpy as np
from transformers.trainer_utils import PredictionOutput
from transformers.trainer_pt_utils import nested_detach
from torch.utils.data import Dataset
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

class DetailedSampleTrainer(CustomSeq2SeqTrainer):
    """
    Extension of CustomSeq2SeqTrainer that saves detailed sample-level predictions
    for later analysis and comparison.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_in_eval = False
        
    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for the model, preserving original inputs.
        """
        # Store original inputs for later reference
        if not hasattr(self, "_original_inputs"):
            self._original_inputs = []
        
        # Only store during evaluation to avoid memory issues during training
        if self.is_in_eval:
            # Deep copy the inputs to avoid reference issues
            self._original_inputs.append({k: v.cpu().clone() if isinstance(v, torch.Tensor) else v 
                                         for k, v in inputs.items()})
        
        return super()._prepare_inputs(inputs)
        
    def evaluate(self, *args, **kwargs):
        """
        Run evaluation and save detailed results.
        """
        # Initialize storage for original inputs
        self._original_inputs = []
        self.is_in_eval = True
        
        # Run the standard evaluation
        result = super().evaluate(*args, **kwargs)
        
        # Reset flag
        self.is_in_eval = False
        
        # Save detailed predictions if we have them
        if hasattr(self, "_last_predictions") and self._last_predictions is not None:
            self.save_detailed_predictions(self._last_predictions)
            
        return result
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        """
        Override prediction_step to capture detailed predictions.
        """
        # Call the original prediction_step method
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **kwargs
        )
        
        # Store detailed prediction information for later processing
        if not prediction_loss_only and generated_tokens is not None:
            if not hasattr(self, "_detailed_predictions"):
                self._detailed_predictions = []
                
            # Store the predictions, inputs, and labels for later processing
            self._detailed_predictions.append({
                "generated": nested_detach(generated_tokens),
                "labels": nested_detach(labels) if labels is not None else None,
                "inputs": {k: nested_detach(v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            })
            
        return loss, generated_tokens, labels
    
    def save_detailed_predictions(self, predictions: PredictionOutput):
        """
        Save detailed, sample-level predictions to file.
        """
        if not self.is_world_process_zero():
            return
            
        # Make sure we have a place to save
        os.makedirs(self.args.output_dir, exist_ok=True)
        output_file = os.path.join(self.args.output_dir, "detailed_predictions.jsonl")
        
        try:
            # Collect all detailed predictions from evaluation
            all_results = []
            
            # Process the detailed predictions collected during evaluation
            if hasattr(self, "_detailed_predictions") and len(self._detailed_predictions) > 0:
                tokenizer = self.tokenizer if hasattr(self, "tokenizer") else getattr(self, "processing_class", None)
                
                if tokenizer is None:
                    print("Warning: No tokenizer available for decoding predictions")
                    return
                
                # Process each batch of predictions
                for i, batch in enumerate(self._detailed_predictions):
                    if batch["generated"] is None:
                        continue
                        
                    # Get the original inputs, generated tokens, and labels
                    input_ids = batch["inputs"].get("input_ids")
                    generated_tokens = batch["generated"]
                    labels = batch["labels"]
                    
                    # Decode the inputs, predictions, and labels
                    decoded_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    
                    if labels is not None:
                        # Replace IGNORE_INDEX with pad_token_id to properly decode
                        labels_for_decode = labels.clone()
                        if hasattr(self, "prepare_labels_for_decode"):
                            labels_for_decode = self.prepare_labels_for_decode(labels_for_decode)
                        decoded_labels = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                    else:
                        decoded_labels = [None] * len(decoded_preds)
                    
                    # Store each example with its detailed info
                    for j in range(len(decoded_inputs)):
                        result = {
                            "prompt": decoded_inputs[j],
                            "prediction": decoded_preds[j],
                            "label": decoded_labels[j] if j < len(decoded_labels) else None,
                            "batch_idx": i,
                            "example_idx": j
                        }
                        all_results.append(result)
            
            # Save all results to a JSONL file
            with open(output_file, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
            print(f"Saved {len(all_results)} detailed predictions to {output_file}")
            
        except Exception as e:
            print(f"Error saving detailed predictions: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear the detailed predictions to free memory
        self._detailed_predictions = [] 