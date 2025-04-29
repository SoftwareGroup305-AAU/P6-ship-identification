import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, ops

from core import YOLO
from dataset import YOLODataset

class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_obj = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targets):
        """
        Args:
            preds: Model output dictionary
            targets: List of tensors [batch_idx, class_idx, x, y, w, h]
                     Coordinates are normalized [0,1]
        Returns:
            Total loss and component losses
        """
        # Initialize losses
        loss_cls = torch.tensor(0., device=next(self.parameters()).device)
        loss_box = torch.tensor(0., device=next(self.parameters()).device)
        loss_obj = torch.tensor(0., device=next(self.parameters()).device)
        
        # Process each prediction scale
        for scale, pred in preds.items():
            # Get grid size
            bs, _, ny, nx = pred['bbox'].shape
            stride = 1.0 / max(ny, nx)  # Approximate stride
            
            # Convert targets to this scale's grid
            scale_targets = self._build_targets(targets, nx, ny, stride)
            
            # Classification loss
            if self.num_classes > 1:
                t = torch.zeros_like(pred['cls'])
                t[scale_targets[..., 0], scale_targets[..., 2], scale_targets[..., 1]] = 1.0
                loss_cls += self.bce_cls(pred['cls'], t)
            
            # Objectness loss
            tobj = torch.zeros_like(pred['bbox'][:, 0:1])
            tobj[scale_targets[..., 0], scale_targets[..., 1]] = 1.0
            loss_obj += self.bce_obj(pred['bbox'][:, 0:self.reg_max], tobj)
            
            # Box regression loss (simplified)
            if scale_targets.shape[0]:
                # Convert predictions to boxes (simplified)
                pred_boxes = self._decode_pred(pred['bbox'], nx, ny, stride)
                # Compute GIoU or MSE loss
                loss_box += self._box_loss(pred_boxes, scale_targets[..., 3:7])
        
        # Weight the losses (these weights are tunable hyperparameters)
        loss = 0.05 * loss_cls + 0.7 * loss_box + 0.3 * loss_obj
        return loss, {'cls': loss_cls, 'box': loss_box, 'obj': loss_obj}
    
    def _build_targets(self, targets, nx, ny, stride):
        """Convert targets to grid coordinates"""
        # This is a simplified version - real implementation would handle:
        # - Multiple anchors
        # - Positive/negative sample balancing
        # - Label smoothing
        # etc.
        scale_targets = []
        for i, t in enumerate(targets):
            if t.shape[0] == 0:
                continue
            # Convert normalized coords to grid coords
            grid_x = (t[:, 2] * nx).long()
            grid_y = (t[:, 3] * ny).long()
            # Filter targets that fall on this grid
            valid = (grid_x >= 0) & (grid_x < nx) & (grid_y >= 0) & (grid_y < ny)
            if valid.any():
                scale_targets.append(torch.stack([
                    i * torch.ones(valid.sum()),  # batch index
                    grid_x[valid],
                    grid_y[valid],
                    t[valid, 2:6]  # original box coords
                ], dim=1))
        return torch.cat(scale_targets, 0) if scale_targets else torch.zeros((0, 6))
    
    def _decode_pred(self, pred, nx, ny, stride):
        """Convert model output to box coordinates"""
        # Simplified - real implementation would use distribution focal loss
        # and integrate over the reg_max bins
        pred = pred.permute(0, 2, 3, 1).reshape(-1, 4, self.reg_max)
        pred = torch.softmax(pred, dim=-1).matmul(torch.linspace(0, self.reg_max-1, self.reg_max).to(pred.device))
        pred = pred.reshape(-1, 4)
        
        # Convert to absolute coordinates
        grid = torch.meshgrid(torch.arange(ny), torch.arange(nx))
        grid = torch.stack(grid, dim=-1).to(pred.device)
        pred[..., :2] = (pred[..., :2] + grid) * stride
        pred[..., 2:] = pred[..., 2:].exp() * stride
        return pred
    
    def _box_loss(self, pred, target):
        """Compute box regression loss (GIoU or MSE)"""
        # Simplified - use MSE for this example
        return torch.mean((pred - target)**2)

def train_yolo(model, dataset, epochs=100, batch_size=16, lr=0.001):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize components
    criterion = YOLOLoss(num_classes=model.head.detect_p3.cls.out_channels)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()  # For mixed precision training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_losses = {'cls': 0.0, 'box': 0.0, 'obj': 0.0}
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Mixed precision forward
            with autocast():
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            running_loss += loss.item()
            for k in loss_dict:
                running_losses[k] += loss_dict[k].item()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} Batch {batch_idx} Loss: {loss.item():.4f}')
        
        # Epoch summary
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}')
        print(f'Component Losses - Cls: {running_losses["cls"]/len(dataloader):.4f} '
              f'Box: {running_losses["box"]/len(dataloader):.4f} '
              f'Obj: {running_losses["obj"]/len(dataloader):.4f}')
    
    return model

def collate_fn(batch):
    """Custom collate function to handle variable numbers of targets"""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

# Example usage:
if __name__ == "__main__":
    # Initialize model
    num_classes = 11 # Example for VOC
    model = YOLO(num_classes=num_classes)
    
    train_images = "yolo/data/train/images/"
    train_labels = "yolo/data/train/labels/"

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(Config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    dataset = YOLODataset(train_images, train_labels, )
    
    # Train the model
    trained_model = train_yolo(model, dataset, epochs=50, batch_size=8)