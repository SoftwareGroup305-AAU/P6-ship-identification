import torch

def generate_targets(raw_preds: torch.Tensor, raw_targets: list):
    B, S, _, C = raw_preds.shape
    target = torch.zeros((B, S, S, ))
    for batch_idx in range(B):
        pass


    
    
    
    
    