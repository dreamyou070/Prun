from torch import nn
import torch

# Name: down_blocks.0.motion_modules.0.proj_in, Module: Linear(in_features=320, out_features=320, bias=True)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_q, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_k, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_v, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.to_out.0, Module: Linear(in_features=320, out_features=320, bias=True)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_q, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_k, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_v, Module: Linear(in_features=320, out_features=320, bias=False)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_out.0, Module: Linear(in_features=320, out_features=320, bias=True)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.0.proj, Module: Linear(in_features=320, out_features=2560, bias=True)
# Name: down_blocks.0.motion_modules.0.transformer_blocks.0.ff.net.2, Module: Linear(in_features=1280, out_features=320, bias=True)
# Name: down_blocks.0.motion_modules.0.proj_out, Module: Linear(in_features=320, out_features=320, bias=True)
                
class MaskObject(nn.Module):

    def __init__(self, ):
        super(MaskObject, self).__init__()

    def set_mask(self, layer_name, dim):
        if 'proj_in' in layer_name:
            self.proj_in_mask = torch.ones(dim)
        elif 'attn1.to_q' in layer_name:
            self.attn1_to_q_mask = torch.ones(dim)
        elif 'attn1.to_k' in layer_name:
            self.attn1_to_k_mask = torch.ones(dim)
        elif 'attn1.to_v' in layer_name:
            self.attn1_to_v_mask = torch.ones(dim)
        elif 'attn1.to_out' in layer_name:
            self.attn1_to_out_mask = torch.ones(dim)
        elif 'attn2.to_q' in layer_name:
            self.attn2_to_q_mask = torch.ones(dim)
        elif 'attn2.to_k' in layer_name:
            self.attn2_to_k_mask = torch.ones(dim)
        elif 'attn2.to_v' in layer_name:
            self.attn2_to_v_mask = torch.ones(dim)
        elif 'attn2.to_out' in layer_name:
            self.attn2_to_out_mask = torch.ones(dim)
        elif 'ff.net.0.proj' in layer_name:
            self.ff_net_0_proj_mask = torch.ones(dim)
        elif 'ff.net.2' in layer_name:
            self.ff_net_2_mask = torch.ones(dim)
        elif 'proj_out' in layer_name:
            self.proj_out_mask = torch.ones(dim)

    def forward(self, input, original_layer, layer_name):

        if layer_name == 'proj_in':
            mask = self.proj_in_mask
        elif layer_name == 'attn1.to_q':
            mask = self.attn1_to_q_mask
        elif layer_name == 'attn1.to_k':
            mask = self.attn1_to_k_mask
        elif layer_name == 'attn1.to_v':
            mask = self.attn1_to_v_mask
        elif layer_name == 'attn1.to_out':
            mask = self.attn1_to_out_mask
        elif layer_name == 'attn2.to_q':
            mask = self.attn2_to_q_mask
        elif layer_name == 'attn2.to_k':
            mask = self.attn2_to_k_mask
        elif layer_name == 'attn2.to_v':
            mask = self.attn2_to_v_mask
        elif layer_name == 'attn2.to_out':
            mask = self.attn2_to_out_mask
        elif layer_name == 'ff.net.0.proj':
            mask = self.ff_net_0_proj_mask
        elif layer_name == 'ff.net.2':
            mask = self.ff_net_2_mask
        elif layer_name == 'proj_out':
            mask = self.proj_out_mask

        original_weight = original_layer.weight
        if original_layer.bias is not None:
            original_bias = original_layer.bias
        # [1] sort mask according to the size (from big to small)
        mask = torch.argsort(mask)
        # [2] weighing
        sparsified_weight = mask.unsqueeze(1).expand_as(original_weight) * original_weight
        output = input * sparsified_weight.T
        # [3] apply bias
        if original_layer.bias is not None:
            sparsified_bias = mask * original_bias
            output = output + sparsified_bias
        return output