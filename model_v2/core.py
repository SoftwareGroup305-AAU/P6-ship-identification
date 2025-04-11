import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))
    

class Bottleneck(nn.Module):
    def __init__(self, num_features, shortcut=True):
        super().__init__()
        self.conv1 = ConvBlock(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBlock(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.use_shortcut = shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_shortcut:
            return x + y
        return y


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True):
        super().__init__()
        self.out_channels = out_channels
        self.split_channels = out_channels // 2
        self.n = n
        
        # Initial 1x1 convolution
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Bottleneck blocks
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.split_channels, shortcut) for _ in range(n)
        ])
        
        # Final 1x1 convolution
        self.conv_final = ConvBlock((n + 2) * self.split_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Initial 1x1 conv
        x = self.conv(x)
        
        # Split the tensor
        x1, x2 = torch.split(x, [self.split_channels, self.split_channels], dim=1)
        # Process through bottlenecks
        outputs = [x1, x2]
        for bottleneck in self.bottlenecks:
            x2 = bottleneck(x2)
            outputs.append(x2)
        
        # Concatenate all outputs
        y = torch.cat(outputs, dim=1)
        
        # Final 1x1 conv
        return self.conv_final(y)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = ConvBlock(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        cat = torch.cat([x, y1, y2, y3], dim=1)
        return self.conv2(cat) 

class Detect(nn.Module):
    def __init__(self, num_classes, in_channels, reg_max=16):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1), 
            ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        )
        # Bbox branch (coordinates + objectness ???)
        self.bbox = nn.Conv2d(in_channels, 4 * reg_max, kernel_size=1, stride=1, padding=0)  
        # Class branch
        self.cls = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=1)  

    def forward(self, x):
        x = self.stem(x)
        return {
            "bbox": self.bbox(x),  # Shape: [B, 4*reg_max, H, W] 
            "cls": self.cls(x)     # Shape: [B, nc, H, W]
        }

class Backbone(nn.Module):
    def __init__(self, depth_multiple=0.33, width_multiple=0.25, max_channels=1024):
        super().__init__()
        # Base channels
        self.channels = {
            'P1': int(min(64, max_channels)*width_multiple),
            'P2': int(min(128, max_channels)*width_multiple),
            'P3': int(min(256, max_channels)*width_multiple),
            'P4': int(min(512, max_channels)*width_multiple),
            'P5': int(min(1024, max_channels)*width_multiple)
        }
        
        # Layer definitions
        self.layers = nn.ModuleList([
            # Stage 0: Initial conv (P1)
            ConvBlock(3, self.channels['P1'], kernel_size=3, stride=2, padding=1),
            
            # Stage 1: Conv (P2)
            ConvBlock(self.channels['P1'], self.channels['P2'], kernel_size=3, stride=2, padding=1),
            
            # Stage 2: C2f (n=3*depth_multiple)
            C2f(self.channels['P2'], self.channels['P2'], n=round(3*depth_multiple) , shortcut=True),
            
            # Stage 3: Conv (P3)
            ConvBlock(self.channels['P2'], self.channels['P3'], kernel_size=3, stride=2, padding=1),
            
            # Stage 4: C2f (n=6*depth_multiple)
            C2f(self.channels['P3'], self.channels['P3'], n=round(6*depth_multiple) , shortcut=True),
            
            # Stage 5: Conv (P5)
            ConvBlock(self.channels['P3'], self.channels['P4'], kernel_size=3, stride=2, padding=1),
            
            # Stage 6: C2f (n=6*depth_multiple)
            C2f(self.channels['P4'], self.channels['P4'], n=round(6*depth_multiple) , shortcut=True),
            
            # Stage 7: Conv (P7)
            ConvBlock(self.channels['P4'], self.channels['P5'], kernel_size=3, stride=2, padding=1),
            
            # Stage 8: Final C2f (n=3*depth_multiple)
            C2f(self.channels['P5'], self.channels['P5'], n=round(3*depth_multiple) , shortcut=True)
        ])
        
        # Output indices for neck connections
        self.out_indices = [4, 6, 8]  # P3, P5, P7 (matches YOLOv8)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return {
            "p3": outputs[0],
            "p5": outputs[1],
            "p7": outputs[2]
        }

class Neck(nn.Module):
    def __init__(self, channels, depth_multiple=0.33):
        super().__init__()
        self.channels = channels
        
        self.sppf = SPPF(channels['p7'])
        self.upsample = nn.Upsample(scale_factor=2)
        
        # Fusion blocks with channel projection
        self.fuse_p1 = C2f(channels['p5'] + channels['p7'], channels['p5'], 
                          n=round(3*depth_multiple), shortcut=False)
        self.fuse_p2 = C2f(channels['p3'] + channels['p5'], channels['p3'],
                          n=round(3*depth_multiple), shortcut=False)
        self.fuse_p3 = C2f(channels['p3'] + channels['p5'], channels['p5'], 
                          n=round(3*depth_multiple), shortcut=False)
        self.fuse_p4 = C2f(channels['p5'] + channels['p7'], channels['p7'], 
                          n=round(3*depth_multiple), shortcut=False)
        # Downsample convs
        self.down_p3 = ConvBlock(channels['p3'], channels['p3'], 
                               kernel_size=3, stride=2, padding=1)
        self.down_p5 = ConvBlock(channels['p5'], channels['p5'],
                               kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        p3, p5, p7 = inputs['p3'], inputs['p5'], inputs['p7']
        
        # Up path
        p7 = self.sppf(p7)
        p7_up = self.upsample(p7)
        p5 = self.fuse_p1(torch.cat([p5, p7_up], dim=1))  # Channels: p5 + p7 → p5
        
        p5_up = self.upsample(p5)
        p3 = self.fuse_p2(torch.cat([p3, p5_up], dim=1))  # Channels: p3 + p5 → p3
        
        # Down path
        p3_down = self.down_p3(p3)
        p5 = self.fuse_p3(torch.cat([p5, p3_down], dim=1)) 
        
        p5_down = self.down_p5(p5)
        p7 = self.fuse_p4(torch.cat([p7, p5_down], dim=1))  
        
        return {'p3': p3, 'p5': p5, 'p7': p7}

class Head(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()        
        self.detect_p3 = Detect(num_classes, channels["p3"])
        self.detect_p5 = Detect(num_classes, channels["p5"])
        self.detect_p7 = Detect(num_classes, channels["p7"])
    def forward(self, x):
        p3, p5, p7 = x["p3"],  x["p5"],  x["p7"]
        return {
            "p3": self.detect_p3(p3),
            "p5": self.detect_p5(p5),
            "p7": self.detect_p7(p7)
        }
    
class YOLO(nn.Module):
    def __init__(self, num_classes, depth_multiple=0.33, width_multiple=0.25, max_channels=1024):
        channels={
                'p3': int(min(256, max_channels) * width_multiple),
                'p5': int(min(512, max_channels) * width_multiple),
                'p7': int(min(1024, max_channels) * width_multiple)
            }
        super().__init__()
        self.backbone = Backbone(depth_multiple, width_multiple, max_channels)
        self.neck = Neck(channels, depth_multiple)
        self.head = Head(channels, num_classes)
    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))