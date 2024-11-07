from torch import nn
import torch
from typing import Optional, Dict, Any
class SimpleAttention(nn.Module):
    """ Insteade using computational cost high attention block,
    use simple convolutional block """

    def __init__(self, dim, layer_name=""):
        super().__init__()
        input_channels = dim
        output_channels = dim
        kernel_size = 3
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._zero_initialize()
        self.layer_name = layer_name

    def _zero_initialize(self):

        self.conv2d.weight.data.zero_()
        self.conv2d.bias.data.zero_()
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: Optional[torch.LongTensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                class_labels: torch.LongTensor = None,
                num_frames: int = 1,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                return_dict: bool = True, ):
        # [1] reshaping
        batch_frames, channel, height, width = hidden_states.shape  # batch_frames = frame
        x = self.conv2d(hidden_states)
        x = self.relu(x)
        x = self.pool(x)
        return x

block_net = SimpleAttention(dim=320)
batch_frames, channel, height, width = 16, 320, 64,64
x = torch.randn(batch_frames, channel, height, width)
output = block_net(x)
print(output.shape)  # torch.Size([16, 320, 32, 64])