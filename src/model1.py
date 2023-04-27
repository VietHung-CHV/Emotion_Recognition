import timm
import torch.nn as nn
from torchsummary import summary

base_model=timm.create_model('resnet50', num_classes=5, pretrained=False)
# print(base_model)

class Model1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            base_model,
            nn.Dropout(0.5),
            nn.Flatten(), 
            nn.Linear(2*2*2048, 32),
            nn.BatchNorm2d(),            
            nn.ReLU(),
            
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.BatchNorm2d(),
            
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, xb):
        return self.model(xb)

    def __repr__(self):
        return f"{self.model}"
    
    def __str__(self):
        summary(self.model, (3, 224, 224))