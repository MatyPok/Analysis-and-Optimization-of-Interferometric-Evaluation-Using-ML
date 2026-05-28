# ------------------------------------------------------------
# PyTorch Models
# ------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetRegressor(nn.Module):
 
 
    """Modified ResNet18 for predicting Zernike coefficients Z4–Z15.
    
    Takes the interferogram image and the total tilt (Z2, Z3) as inputs.
    The tilt is concatenated with the CNN features before the regression
    head, allowing the model to resolve the cos(phase) sign ambiguity.
    """
    
    def __init__(self, num_outputs: int = 12, pretrained: bool = True, in_channels: int = 1):
        super().__init__()
    
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
    
        # Replace first conv to accept 1- or 2-channel input instead of 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))
        
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        # Regression head: 512 (ResNet features) + 2 (tilt z2, z3)
        self.fc = nn.Sequential(
        nn.Linear(512 + 2, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_outputs),
        )
        
    def forward(self, x: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Concatenate tilt info so the model knows the fringe direction
        x = torch.cat([x, tilt], dim=1)
        x = self.fc(x)
        return x
