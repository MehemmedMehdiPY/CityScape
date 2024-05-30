import torch
from torch import nn
import segmentation_models_pytorch as smp
from typing import Optional, Tuple

class Unet(nn.Module):
    def __init__(self, encoder_name: Optional[str] = 'resnet18', encoder_depth: Optional[int] = 5, encoder_weights: Optional[str] = None, 
                 decoder_channels: Optional[Tuple] = (256, 128, 64, 32, 16), in_channels: Optional[int] = 3, pretrained_encoder: bool = False, 
                 encoder_path: str = None, num_classes: Optional[int] = 1, device: Optional[str] = None) -> None:
        
        if pretrained_encoder and encoder_path is None:
            raise ValueError('encoder_path must be input while pretrained_encoder=True')
        
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.encoder_path = encoder_path
        self.num_classes = num_classes

        self.device = ('cpu' if device is None
                  else device)
    
        self.unet = smp.Unet(encoder_name=self.encoder_name, encoder_depth=self.encoder_depth, encoder_weights=self.encoder_weights, 
                              decoder_channels=self.decoder_channels, in_channels=self.in_channels, classes=self.num_classes)\
                                .to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

        if pretrained_encoder:
            checkpoints = torch.load(self.encoder_path)
            self.unet.encoder.load_state_dict(checkpoints)

    def forward(self, x):
        x = self.unet(x)
        x = self.softmax(x)
        return x
