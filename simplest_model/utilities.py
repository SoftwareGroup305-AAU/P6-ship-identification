import torch

def generate_targets(raw_preds: torch.Tensor, raw_targets: list, number_of_bboxes):
    """
    Args:
        raw_preds: [B, S, S, b*5+c] – predicted tensor
        raw_targets: list of B elements, each containing a list of [class, x, y, w, h]
                    where x, y, w, h are normalized to [0,1]
    Returns:
        torch.Tensor: A tensor of shape (B, S, S, B*5 + C) representing the training targets.
    """
    B, S, _, C = raw_preds.shape
    targets = torch.zeros((B, S, S, C), device=raw_preds.device)


    for batch_idx, targets_batch in enumerate(raw_targets):
        for box in targets_batch:
            class_idx, x, y, w, h = box

            grid_x = int(x * S)
            grid_y = int(y * S)

            x_offset = x * S - grid_x
            y_offset = y * S - grid_y

            # Box coordinates go into the first bbox slot (index 0)
            targets[batch_idx, grid_y, grid_x, 0:5] = torch.tensor(
                [x_offset, y_offset, w, h, 1], device=raw_preds.device
            )

            # Class probabilities (one-hot)
            targets[batch_idx, grid_y, grid_x, number_of_bboxes*5+int(class_idx)] = 1
    return targets


                



    
    
    
    
    