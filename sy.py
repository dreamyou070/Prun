import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class TemporalWindowAttention(nn.Module):
    def __init__(self, num_frames, embed_size, window_size, heads):
        super(TemporalWindowAttention, self).__init__()
        self.num_frames = num_frames
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_size, heads)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 16,
                frames_mask: Optional[torch.LongTensor] = None):
        
       # :param hidden_states: (batch_size, num_frames, channels, height, width)
       # :param frames_mask: (batch_size, seq_len) or None
       # :return: attention output, attention weights
        
        residual = hidden_states

        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4) # batch, channel, frame, height, width
        height = hidden_states.shape[3]
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        # [2] 슬라이딩 윈도우 분할
        assert num_frames % self.window_size == 0, "num_frames must be divisible by window_size"
        attn_outputs = []
        for start_idx in range(0, num_frames, self.window_size):
            end_idx = min(start_idx + self.window_size, num_frames)
            window_x = hidden_states[:, start_idx:end_idx, :] # batch, window_size, channel
            print(f"window_x shape: {window_x.shape}")
            # 각 윈도우 내에서 어텐션 계산
            attn_out, _ = self.attn(window_x, window_x, window_x)
            attn_outputs.append(attn_out)
        attn_output = torch.cat(attn_outputs, dim=1) # (B, num_frames, C)
        # reshape to original
        attn_output = attn_output.reshape(batch_size, height, width, num_frames,-1)  # [batch, height, width, num_frames, channel]
        attn_output = attn_output.permute(0, 3, 4, 1, 2).contiguous()  # [batch, num_frames, channel, height, width]
        attn_output = attn_output.reshape(batch_frames, channel, height, width)  # [batch_frame, channel, height, width]
        output = attn_output + residual
        return output

