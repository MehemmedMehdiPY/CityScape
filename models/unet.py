from torch import nn
import segmentation_models_pytorch as smp
from typing import Optional, Tuple

class Unet(nn.Module):
    def __init__(self, encoder_name: Optional[str] = 'resnet18', encoder_depth: Optional[int] = 5, encoder_weights: Optional[str] = None, 
                 decoder_channels: Optional[Tuple] = (256, 128, 64, 32, 16), in_channels: Optional[int] = 3,
                 classes: Optional[int] = 1, device: Optional[str] = None) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_depth = encoder_depth
        self.encoder_weights = encoder_weights
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.classes = classes

        self.device = ('cpu' if device is None
                  else device)
        
        self.model = smp.Unet(encoder_name=self.encoder_name, encoder_depth=self.encoder_depth, encoder_weights=self.encoder_weights, 
                              decoder_channels=self.decoder_channels, in_channels=self.in_channels, classes=self.classes)\
                                .to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x
