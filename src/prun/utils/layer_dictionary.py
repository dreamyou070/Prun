layers_basic = ['down_blocks_0_motion_modules_0_transformer_blocks_0',
                'down_blocks_0_motion_modules_1_transformer_blocks_0',
                'down_blocks_1_motion_modules_0_transformer_blocks_0',
                'down_blocks_1_motion_modules_1_transformer_blocks_0',
                'down_blocks_2_motion_modules_0_transformer_blocks_0',
                'down_blocks_2_motion_modules_1_transformer_blocks_0',
                'down_blocks_3_motion_modules_0_transformer_blocks_0',
                'down_blocks_3_motion_modules_1_transformer_blocks_0',
                'up_blocks_0_motion_modules_0_transformer_blocks_0','up_blocks_0_motion_modules_1_transformer_blocks_0',
                'up_blocks_0_motion_modules_2_transformer_blocks_0','up_blocks_1_motion_modules_0_transformer_blocks_0',
                'up_blocks_1_motion_modules_1_transformer_blocks_0','up_blocks_1_motion_modules_2_transformer_blocks_0',
                'up_blocks_2_motion_modules_0_transformer_blocks_0','up_blocks_2_motion_modules_1_transformer_blocks_0',
                'up_blocks_2_motion_modules_2_transformer_blocks_0','up_blocks_3_motion_modules_0_transformer_blocks_0',
                'up_blocks_3_motion_modules_1_transformer_blocks_0',
                'up_blocks_3_motion_modules_2_transformer_blocks_0',
                'mid_block_motion_modules_0_transformer_blocks_0']

layers = ['down_blocks_0_motion_modules_0_transformer_blocks_0_attn1','down_blocks_0_motion_modules_0_transformer_blocks_0_attn2',
          'down_blocks_0_motion_modules_1_transformer_blocks_0_attn1','down_blocks_0_motion_modules_1_transformer_blocks_0_attn1',
          'down_blocks_1_motion_modules_0_transformer_blocks_0_attn1','down_blocks_1_motion_modules_0_transformer_blocks_0_attn1',
          'down_blocks_1_motion_modules_1_transformer_blocks_0_attn1','down_blocks_1_motion_modules_1_transformer_blocks_0_attn1',
          'down_blocks_2_motion_modules_0_transformer_blocks_0_attn1','down_blocks_2_motion_modules_0_transformer_blocks_0_attn1',
          'down_blocks_2_motion_modules_1_transformer_blocks_0_attn1','down_blocks_2_motion_modules_1_transformer_blocks_0_attn1',
          'down_blocks_3_motion_modules_0_transformer_blocks_0_attn1','down_blocks_3_motion_modules_0_transformer_blocks_0_attn1',
          'down_blocks_3_motion_modules_1_transformer_blocks_0_attn1','down_blocks_3_motion_modules_1_transformer_blocks_0_attn1',
          'mid_block_motion_modules_0_transformer_blocks_0_attn1','mid_block_motion_modules_0_transformer_blocks_0_attn1',
          'up_blocks_0_motion_modules_0_transformer_blocks_0_attn1','up_blocks_0_motion_modules_0_transformer_blocks_0_attn1',
          'up_blocks_0_motion_modules_1_transformer_blocks_0_attn1','up_blocks_0_motion_modules_1_transformer_blocks_0_attn1',
          'up_blocks_0_motion_modules_2_transformer_blocks_0_attn1','up_blocks_0_motion_modules_2_transformer_blocks_0_attn1',
          'up_blocks_1_motion_modules_0_transformer_blocks_0_attn1','up_blocks_1_motion_modules_0_transformer_blocks_0_attn1',
          'up_blocks_1_motion_modules_1_transformer_blocks_0_attn1','up_blocks_1_motion_modules_1_transformer_blocks_0_attn1',
          'up_blocks_1_motion_modules_2_transformer_blocks_0_attn1','up_blocks_1_motion_modules_2_transformer_blocks_0_attn1',
          'up_blocks_2_motion_modules_0_transformer_blocks_0_attn1','up_blocks_2_motion_modules_0_transformer_blocks_0_attn1',
          'up_blocks_2_motion_modules_1_transformer_blocks_0_attn1','up_blocks_2_motion_modules_1_transformer_blocks_0_attn1',
          'up_blocks_2_motion_modules_2_transformer_blocks_0_attn1','up_blocks_2_motion_modules_2_transformer_blocks_0_attn1',
          'up_blocks_3_motion_modules_0_transformer_blocks_0_attn1','up_blocks_3_motion_modules_0_transformer_blocks_0_attn1',
          'up_blocks_3_motion_modules_1_transformer_blocks_0_attn1','up_blocks_3_motion_modules_1_transformer_blocks_0_attn1',
          'up_blocks_3_motion_modules_2_transformer_blocks_0_attn1', 'up_blocks_3_motion_modules_2_transformer_blocks_0_attn1',]

layer_dict = {0: 'down_blocks_0_motion_modules_0',
              1: 'down_blocks_0_motion_modules_1',
              2: 'down_blocks_1_motion_modules_0',
              3: 'down_blocks_1_motion_modules_1',
              4: 'down_blocks_2_motion_modules_0',
              5: 'down_blocks_2_motion_modules_1',
              6: 'down_blocks_3_motion_modules_0',
              7: 'down_blocks_3_motion_modules_1',
              8: 'mid_block_motion_modules_0',
              9: 'up_blocks_0_motion_modules_0',
              10: 'up_blocks_0_motion_modules_1',
              11: 'up_blocks_0_motion_modules_2',
              12: 'up_blocks_1_motion_modules_0',
              13: 'up_blocks_1_motion_modules_1',
              14: 'up_blocks_1_motion_modules_2',
              15: 'up_blocks_2_motion_modules_0',
              16: 'up_blocks_2_motion_modules_1',
              17: 'up_blocks_2_motion_modules_2',
              18: 'up_blocks_3_motion_modules_0',
              19: 'up_blocks_3_motion_modules_1',
              20: 'up_blocks_3_motion_modules_2'}
layer_dict_dot = {0: 'down_blocks.0.motion_modules.0',
                  1: 'down_blocks.0.motion_modules.1',
                  2: 'down_blocks.1.motion_modules.0',
                  3: 'down_blocks.1.motion_modules.1',
                  4: 'down_blocks.2.motion_modules.0',
                  5: 'down_blocks.2.motion_modules.1',
                  6: 'down_blocks.3.motion_modules.0',
                  7: 'down_blocks.3.motion_modules.1',
                  8: 'mid_block.motion_modules.0',
                    9: 'up_blocks.0.motion_modules.0',
                    10: 'up_blocks.0.motion_modules.1',
                    11: 'up_blocks.0.motion_modules.2',
                    12: 'up_blocks.1.motion_modules.0',
                    13: 'up_blocks.1.motion_modules.1',
                    14: 'up_blocks.1.motion_modules.2',
                    15: 'up_blocks.2.motion_modules.0',
                    16: 'up_blocks.2.motion_modules.1',
                    17: 'up_blocks.2.motion_modules.2',
                    18: 'up_blocks.3.motion_modules.0',
                    19: 'up_blocks.3.motion_modules.1',
                    20: 'up_blocks.3.motion_modules.2'}


layer_dict_short = {0: 'down_0_0', 1: 'down_0_1',
                    2: 'down_1_0', 3: 'down_1_1',
                    4: 'down_2_0', 5: 'down_2_1',
                    6: 'down_3_0', 7: 'down_3_1',
                    8: 'mid',
                    9: 'up_0_0', 10: 'up_0_1', 11: 'up_0_2',
                   12: 'up_1_0', 13: 'up_1_1', 14: 'up_1_2',
                   15: 'up_2_0', 16: 'up_2_1', 17: 'up_2_2',
                   18: 'up_3_0', 19: 'up_3_1', 20: 'up_3_2',}


def find_layer_name (skip_layers) :
    target_layers = []
    target_layers_dot = []
    # up_3_0
    for layer in skip_layers :
        # find key using value
        target_key = [key for key, value in layer_dict_short.items() if value == layer] # [18]
        if len(target_key) != 0 :
            for k in target_key :
                layer = layer_dict[k]
                target_layers.append(layer)
                layer_dot = layer_dict_dot[k]
                target_layers_dot.append(layer_dot)
    return target_layers, target_layers_dot

def find_next_layer_name(skip_layers) :
    block_indexs = []
    for block in skip_layers:
        block_idx = [idx for idx, layer in layer_dict_short.items() if layer == block]
        block_indexs.extend(block_idx)
    block_indexs = sorted(block_indexs)
    # not sequence indexs

    def find_next_step(block_indexs, index):
        next_index = index + 1
        if next_index in block_indexs:  # next index is removing
            return find_next_step(block_indexs, next_index)
        else:
            return next_index
    trgs = []
    for idx in block_indexs:
        trg = find_next_step(block_indexs, idx)
        trgs.append(trg)
    trgs = set(trgs)  # found trgs
    saving_layer_names = []
    for trg in trgs:
        trg_layer_name = layer_dict[trg]
        saving_layer_names.append(trg_layer_name)
    return saving_layer_names