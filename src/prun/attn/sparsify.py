import torch
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import os

def sparsify_layer_index(original_layer, # linear_layer
                         pruning_indexs,
                         in_channel_reduce=False, channelwise_sparsity=True, redistribution = False, layer_name = None):
    def sparsify_linear_layer(original_layer,
                              pruning_indexs,
                              in_channel_reduce=False,
                              channelwise_sparsity=True,
                              redistribution=False):
        original_weight = original_layer.weight
        output_dim = original_weight.shape[0]  # 320
        input_dim = original_weight.shape[1] # 1280
        # [1] make new layer 
        sparsified_layer = nn.Linear(input_dim, output_dim, bias=(original_layer.bias is not None))
        sparse_weight = original_weight.cpu().detach().numpy()
        sparse_weight[pruning_indexs] = 0
        sparse_weight = torch.tensor(sparse_weight)
        sparsified_layer.weight = nn.Parameter(sparse_weight)
        if original_layer.bias is not None:
            original_bias = original_layer.bias
            sparse_bias = original_bias.cpu().detach().numpy()
            sparse_bias[pruning_indexs] = 0
            # sparse_bias = sparse_bias[remain_index]
            sparsified_layer.bias = nn.Parameter(torch.tensor(sparse_bias))
        return sparsified_layer
    
    if original_layer.__class__.__name__ == 'Linear':
        original_layer = sparsify_linear_layer(original_layer,
                                               pruning_indexs,
                                               in_channel_reduce,
                                               channelwise_sparsity,
                                               redistribution)
    return original_layer

def sparsify_layer(original_layer, pruning_ratio=.5, in_channel_reduce=False, channelwise_sparsity=True, redistribution = False, layer_name = None):
    
    # in_channel_reduce = True
    def sparsify_linear_layer(original_layer, pruning_ratio=0.5,
                              in_channel_reduce=False,
                              channelwise_sparsity=True,
                              redistribution=False):
        original_weight = original_layer.weight
        input_dim = original_weight.shape[0]
        output_dim = original_weight.shape[1]
        
        if channelwise_sparsity:
            sparse_weight = original_weight.cpu().detach().numpy()
            row_magnitude = np.linalg.norm(sparse_weight, axis=1)
            pruning_dim = int(output_dim * pruning_ratio)
            
            pruning_index = np.argsort(row_magnitude)[:pruning_dim]  
            # this can be []          
            sparse_weight[pruning_index] = 0
            sparse_weight = torch.tensor(sparse_weight)            
            sparsified_layer = nn.Linear(input_dim, output_dim, bias=(original_layer.bias is not None))
            sparsified_layer.weight = nn.Parameter(sparse_weight)            
            if original_layer.bias is not None:
                original_bias = original_layer.bias
                sparse_bias = original_bias.cpu().detach().numpy()
                sparse_bias[pruning_index] = 0
                sparsified_layer.bias = nn.Parameter(torch.tensor(sparse_bias))            
            return sparsified_layer
        else :
            # layer_name
            sorted_idx_list = []
            base = f'/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3_save'
            target_file_name = os.path.join(base, f'{layer_name}_sorted_index_importance.txt')    
            sub_target_file_name = None
            if 'to_q' in target_file_name :
                sub_target_file_name = target_file_name.replace('to_q', 'to_k')
            if 'to_k' in target_file_name :
                sub_target_file_name = target_file_name.replace('to_k', 'to_q')
            with open(target_file_name, 'r') as f:
                content = f.readlines()
            content = content[1:]
            for i, line in enumerate(content) :
                idx, score = line.split(' : ')
                idx = int(idx)
                score = float(score)
                if score < output_dim / 6:
                    sorted_idx_list.append(idx)
            if sub_target_file_name is not None :
                with open(sub_target_file_name, 'r') as f:
                    content = f.readlines()
                content = content[1:]
                for i, line in enumerate(content) :
                    idx, score = line.split(' : ')
                    idx = int(idx)
                    score = float(score)
                    if score < output_dim / 6:
                        sorted_idx_list.append(idx)
                    
            pruning_index = torch.tensor(sorted_idx_list)
            remain_index = torch.tensor([i for i in range(output_dim) if i not in pruning_index])
            
            # [1] make new layer
            new_output_dim = len(remain_index)
            sparsified_layer = nn.Linear(input_dim, new_output_dim, bias=(original_layer.bias is not None))
            # [2] set the weight
            # [1] erase output dimension
            sparse_weight = original_weight.cpu().detach().numpy()
            # get only the remain_index
            sparse_weight = sparse_weight[remain_index,:]
            #sparse_weight[pruning_index] = 0
            #sparse_weight = torch.tensor(sparse_weight)
            #sparsified_layer = nn.Linear(input_dim, output_dim, bias=(original_layer.bias is not None))
            sparse_weight = torch.tensor(sparse_weight)
            sparsified_layer.weight = nn.Parameter(sparse_weight)
            if original_layer.bias is not None:
                original_bias = original_layer.bias
                sparse_bias = original_bias.cpu().detach().numpy()
                #sparse_bias[pruning_index] = 0
                sparse_bias = sparse_bias[remain_index]
                sparsified_layer.bias = nn.Parameter(torch.tensor(sparse_bias))
            return sparsified_layer, output_dim, pruning_index, remain_index        
    if original_layer.__class__.__name__ == 'Linear':
        # True
        original_layer = sparsify_linear_layer(original_layer, pruning_ratio, in_channel_reduce,
                                               channelwise_sparsity, redistribution)
        return original_layer
    else:
        return original_layer
    
def sparsify_layer_unordered(original_layer, pruning_dim=30, in_channel_reduce=False):
    # 1. check weather the layer is linear or norm layer
    # 2. if linear, then apply the sparsification
    # 3. if norm, then pass the layer
    # 4. return the layer
    def sparsify_linear_layer(original_layer, pruning_dim=30, in_channel_reduce=False):
        
        # in_channel_reduce = True
        # reduce output dimension
        # every column weight, erase smallest pruning_dim values
        # if possible do not use numpy but use torch
        
        original_weight = original_layer.weight
        input_dim = original_weight.shape[0]
        output_dim = original_weight.shape[1]
        final_output = output_dim - pruning_dim
        if original_layer.bias is not None:
            sparsified_layer = torch.nn.Linear(input_dim, output_dim - pruning_dim, bias=True)
        else:
            sparsified_layer = torch.nn.Linear(input_dim, output_dim - pruning_dim, bias=False)
        # from the last, get pruning dum
        sparse_weight = original_weight.cpu().detach()[:final_output, :]  # erase 200 dim

        sparsified_layer.weight = torch.nn.Parameter(sparse_weight)
        if original_layer.bias is not None:
            original_bias = original_layer.bias
            sparse_bias = original_bias.cpu().detach()[:final_output]
            # sparse_bias = torch.tensor(sparse_bias)
            sparsified_layer.bias = torch.nn.Parameter(sparse_bias)

        return sparsified_layer

    if original_layer.__class__.__name__ == 'Linear':
        sparsified_layer = sparsify_linear_layer(original_layer, pruning_dim, in_channel_reduce)
        return sparsified_layer
    else:
        return original_layer


def pruned_output(input_tensor: torch.Tensor,
                  original_layer,
                  pruning_ratio=0.5, # pruning_ratio
                  in_channel_reduce=False,
                  unorder=False,
                  channelwise_sparsity=True,
                  redistribution=False,
                  layer_name = None,
                  base_save_dir = None):
    
    channelwise_sparsity = False
    redistribution = True
    main_device = input_tensor.device
    pruned_layer, output_dim, pruning_index, remain_index = sparsify_layer(original_layer, pruning_ratio, in_channel_reduce,
                                                                           channelwise_sparsity=channelwise_sparsity,
                                                                           redistribution=redistribution, layer_name=layer_name)
    if channelwise_sparsity :
        pruned_layer = pruned_layer.to(main_device)
        # 2. make pruned output and recover prunded dim
        # sparse output dim should be same as pruned output dim
        pruned_output = pruned_layer(input_tensor)
        return pruned_output
    else :
        pruned_layer = pruned_layer.to(main_device)
        # 2. make pruned output and recover prunded dim
        # sparse output dim should be same as pruned output dim
        pruned_output = pruned_layer(input_tensor)
        common_shape = pruned_output.shape[:-1]
        common_shape = [int(c.item()) if type(c) == torch.Tensor else c for c in common_shape ]
        common_shape.append(output_dim)
        common_shape = tuple(common_shape)
        org = torch.zeros(common_shape).to(main_device)
        org[..., remain_index] = pruned_output
        #return pruned_output
        return org
        """
        #
        
        #
        pruned_layer = pruned_layer.to(main_device)
        pruned_output = pruned_layer(input_tensor)
        # select the smallest index and set pruning_index
        # [1] get the smallest index (axis = -1)
        original_dim = pruned_output.dim()
        target_dim = [int(i) for i in range(original_dim-1)]
        target_dim = tuple(target_dim)
        dimensionwise_magnitude = torch.norm(pruned_output, dim=target_dim)
        sorted_idx = torch.argsort(dimensionwise_magnitude)
        #layer_save_dir = os.path.join(base_save_dir, layer_name)
        #os.makedirs(layer_save_dir, exist_ok=True)
        # save to layer_name_file
        #print(f'sorted_inx = {sorted_idx}')
        sorted_index_list = sorted_idx.cpu().detach().numpy().tolist()
        with open(os.path.join(base_save_dir, f'{layer_name}_sorted_index.txt'), 'a') as f:
            f.write(str(sorted_index_list))
            f.write(f' \n')
        sorted_idx_list = []
        base = f'/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3_save'
        target_file_name = os.path.join(base, f'{layer_name}_importance.txt')    
        with open(target_file_name, 'w') as f:
            content = f.readlines()
        content = content[1:]
        for line in content :
            idx, score = line.split(' : ')
            idx = int(idx)
            score = float(score)
            if score < 50 :
                sorted_index_list.append(idx)
        pruning_idx = torch.tensor(sorted_index_list)
        """
                
            
            
            
            
            
            
            
        # save sorted_index to 
        #pruning_idx = sorted_idx[:int(pruning_ratio * pruned_output.shape[-1])]
        # [3] set the pruning index to zero
        pruned_output[..., pruning_idx] = 0
        return pruned_output
        
        
        
        # erase dim
    
    # this can be shorted ...
    """
    if unorder:
        # add remain output
        if pruned_output.dim() == 2:
            b = pruned_output.shape[0]
            remain_output = torch.zeros((b, pruning_dim)).to(main_device)
        elif pruned_output.dim() == 3:
            b = pruned_output.shape[0]
            c = pruned_output.shape[1]
            remain_output = torch.zeros((b, c, pruning_dim)).to(main_device)
        elif pruned_output.dim() == 4:
            b = pruned_output.shape[0]
            c = pruned_output.shape[1]
            h = pruned_output.shape[2]
            remain_output = torch.zeros((b, c, h, pruning_dim)).to(main_device)
        pruned_output = torch.cat([pruned_output, remain_output], dim=-1)
    """
    # sparse_output have same shpae but the last dim should be pruned_dim
    return pruned_output
"""
def main() :
    print(f' step1. make pruned layer')
    input_tensor = torch.randn((2,3,5))
    original_layer = nn.Linear(5, 4)
    output_tensor = pruned_output(input_tensor = input_tensor,
                                original_layer=original_layer,
                                pruning_ratio=0.5,
                                in_channel_reduce=False,
                                unorder=False) # half beccome zero (that means two dimmnsion become zero)
    print(f'input_tensor shape = {input_tensor.shape}')
    print(f'output_tensor shape = {output_tensor.shape}')
    print(f'output_tensor = {output_tensor}')

if __name__ == '__main__' :
    main()
"""
