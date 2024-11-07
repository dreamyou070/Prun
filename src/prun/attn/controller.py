import torch
from einops import rearrange
import math
from torch import nn

class AttentionBase:

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after stepdef b
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        # in forward, final attentio is
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0



class MotionControl(nn.Module,AttentionBase):
    
    def __init__(self,
                 is_teacher=False,
                 train=True,
                 repeat_test=False,
                 first_frame_iter=False,
                 frame_num=16,
                 guidance_scale=1,
                 window_attention=False,
                 skip_layers=[],
                 save_layers=[],
                 layerwise=True,
                 do_attention_map_check=False,
                 attention_reshaping_test=False,
                 batch_size=1,
                 do_save_attention_map=False,
                 output_matching=False,
                 do_first_frame_except=False,
                 spatial_criterion=32 * 3,
                 cross_criterion=32 * 3,
                 self_attn_time_criterion=2,
                 cross_attn_time_criterion=2,
                 crossattn_matching=False,
                 crossattn_layerwise=False,
                 temporal_longing=False,
                 do_masactrl_forward=False,
                 crossmatching_res=[8, 16, 32, 64],
                 only_key_change=False,
                 only_value_change=False,
                 temporal_firstframe_attention = False,
                 add_qkv = False,
                 optimization_target_layer = None,
                 inference_only_front_motion_module = False,
                 inference_only_post_motion_module = False,
                 remain_head = [],
                 front_step = 1,
                 criteria = None,
                 control_dim_dix=0,
                 pruned_unorder = False,
                 self_attn_pruned = False,
                 do_single_attention = False,
                 base_save_dir = None,
                 target_timestep = 0,
                 feature_reuse_policy={},
                 block_skip_policy={},):

        nn.Module.__init__(self)
        # [1] layer architecture
        self.motion_block_num = 0
        self.total_self_attn_layer = 21
        self.total_cross_atn_layer = 21
        self.total_temporal_attn_layer = 42

        # [2] state
        
        self.is_teacher = is_teacher
        self.train = train
        self.repeat_test = repeat_test
        self.do_attention_map_check = do_attention_map_check
        self.attention_reshaping_test = attention_reshaping_test
        self.layerwise = layerwise
        self.first_frame_iter = first_frame_iter
        self.crossattn_layerwise = crossattn_layerwise

        # [3] inference related
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.window_attention = window_attention
        self.frame_num = frame_num
        self.attnmap_dict = {}
        self.timestep = 0
        
        self.save_layers = save_layers
        self.skip_layers = skip_layers
        self.layerwise_hidden_dict = {}
        self.classifier_guidance = True if guidance_scale > 1 else False
        if train:
            self.classifier_guidance = False
        self.do_save_attention_map = do_save_attention_map
        self.output_matching = output_matching
        self.do_first_frame_except = do_first_frame_except
        
        # [4] checking
        self.spatial_iter = 0
        self.cross_iter = 0
        
        self.spatial_criterion = spatial_criterion
        self.cross_criterion = cross_criterion
        self.crossattn_matching = crossattn_matching
        self.temporal_longing = temporal_longing
        self.do_masactrl_forward = do_masactrl_forward
        self.time_idx = -1
        self.cross_attn_time_criterion = cross_attn_time_criterion
        self.self_attn_time_criterion = self_attn_time_criterion
        self.crossmatching_res = crossmatching_res
        self.only_key_change = only_key_change
        self.only_value_change = only_value_change
        self.temporal_firstframe_attention = temporal_firstframe_attention 
        
        self.optimization_target_layer = optimization_target_layer
        
        self.inference_only_front_motion_module = inference_only_front_motion_module
        
        #self.inference_only_post_motion_module = inference_only_post_motion_module
        #self.remain_head = remain_head
        #self.front_step  = front_step 
        self.add_qkv = add_qkv
        self.criteria = criteria
        self.control_dim_dix = control_dim_dix
        self.pruning_ratio_dict = {}
        self.pruned_unorder = pruned_unorder
        self.self_attn_pruned = self_attn_pruned
        self.do_single_attention = do_single_attention
        self.base_save_dir = base_save_dir
        self.target_timestep = target_timestep
        self.target_time = target_timestep
        self.cached_motion_states = {}
        self.feature_reuse_policy = feature_reuse_policy
        self.block_skip_policy = block_skip_policy

    def timestepwise(self):
        if self.cross_iter % 16 == 0:
            self.time_idx += 1

            self.spatial_iter = 0  # spatial position
            self.cross_iter = 0

    def cache_hidden_states(self, tensor, layer_name):
        self.cached_motion_states[layer_name] = tensor

    def check_repeat(self, layer_name, is_cross_attn):
        repeat = False
        if is_cross_attn:
            if self.crossattn_layerwise:
                # when there is no temporal attn
                #
                for skip_layer in self.skip_layers:
                    if 'down' in skip_layer or 'up' in skip_layer:
                        position = skip_layer.split('_')[0]
                        block_idx = skip_layer.split('_')[2]
                        layer_idx = skip_layer.split('_')[-1]
                        spatial_name = f'{position}_blocks_{block_idx}_attentions_{layer_idx}'
                    else:
                        spatial_name = 'mid_block_attentions_0_transformer_blocks_0'
                    if spatial_name in layer_name:
                        repeat = True
                        break
            else:
                repeat = True
        if not is_cross_attn:
            if self.layerwise:
                for skip_layer in self.skip_layers:
                    if 'down' in skip_layer or 'up' in skip_layer:
                        position = skip_layer.split('_')[0]
                        block_idx = skip_layer.split('_')[2]
                        layer_idx = skip_layer.split('_')[-1]
                        spatial_name = f'{position}_blocks_{block_idx}_attentions_{layer_idx}'
                    else:
                        spatial_name = 'mid_block_attentions_0_transformer_blocks_0'
                    if spatial_name in layer_name:
                        repeat = True
                        break
            else:
                repeat = True
        return repeat
    
    def set_pruning_ratio(self, layer_name, ratio) :
        # if pruing_ratio = 1
        # it means all prun        
        self.pruning_ratio_dict[layer_name] = ratio
        # ratio

    def save_hidden_states(self, hidden_states=None, layer_name=None):
        #if layer_name not in self.layerwise_hidden_dict:
        #    self.layerwise_hidden_dict[layer_name] = {}
        #self.layerwise_hidden_dict[layer_name][self.timestep] = hidden_states
        self.layerwise_hidden_dict[layer_name] = hidden_states

    def save_attention_map(self, attn_map, layer_name):
        if layer_name not in self.attnmap_dict:
            self.attnmap_dict[layer_name] = []
        self.attnmap_dict[layer_name].append(attn_map)

    def set_timestep(self, timestep):
        self.timestep = timestep

    # motion_block_num = 1 부터 시작하고, 21까지 되면 다시 0 이 된다.
    # editor.timestep
    def time_resetting(self) :
        self.motion_block_num += 1        
        if self.motion_block_num % 21 == 0 :
            self.motion_block_num = 0         
        if self.motion_block_num == 1 :
            self.timestep += 1

    def repeat_attention(self, q, k, v, head_num, head_dim, batch_size, layer_name):
        # print(f'self attention {layer_name} repeat')
        kc, ku = torch.chunk(k, chunks=2, dim=0)
        vc, vu = torch.chunk(v, chunks=2, dim=0)
        if self.first_frame_iter:

            if self.only_value_change:
                first_vc, first_vu = vc[0, :, :, :].clone(), vu[0, :, :, :].clone()
                vc[1:, :, :, :] = first_vc
                vu[1:, :, :, :] = first_vu

            if self.only_key_change:

                first_kc, first_ku = kc[0, :, :, :].clone(), ku[0, :, :, :].clone()
                kc[1:, :, :, :] = first_kc
                ku[1:, :, :, :] = first_ku

            else:

                first_kc, first_ku = kc[0, :, :, :].clone(), ku[0, :, :, :].clone()
                kc[1:, :, :, :] = first_kc
                ku[1:, :, :, :] = first_ku

                first_vc, first_vu = vc[0, :, :, :].clone(), vu[0, :, :, :].clone()
                vc[1:, :, :, :] = first_vc
                vu[1:, :, :, :] = first_vu
        else:
            # [1]
            kc[1:, :, :, :] = kc[:-1, :, :, :].clone()
            ku[1:, :, :, :] = ku[:-1, :, :, :].clone()
            # [2]
            if not self.only_key_change:
                vc[1:, :, :, :] = vc[:-1, :, :, :].clone()
                vu[1:, :, :, :] = vu[:-1, :, :, :].clone()

        k = torch.cat([kc, ku], dim=0)
        v = torch.cat([vc, vu], dim=0)

        # [3]
        dropout_p = 0.0
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
        attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [batch*pixel_num/head/frame/frame]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight,
                                    dim=-1)  # [batch*pixel_num,head, frame, frame] # last frame is only one
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        """
        if self.do_save_attention_map :
            if layer_name not in self.attnmap_dict :
                self.attnmap_dict[layer_name] = []
            self.attnmap_dict[layer_name].append(attn_weight)
        """
        # [4]
        hidden_states = attn_weight @ v  # [pixel_num, head, frame, dim]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, head_num * head_dim)  # [batch*pixel_num, frame, dim]
        return hidden_states

    def masactrl_attention(self, q, k, v):
        # q = [batch*frame, head, pixel, dim]
        # k = [batch*frame, head, pixel, dim]
        # v = [batch*frame, head, pixel, dim]
        # [1] pixel should be attention
        # [2] not attention should be same
        # q = [frame_num, head, pixel_num, dim]
        frame_num = self.frame_num
        batch_size = q.shape[0]
        q = rearrange(q, 'f h p d -> h f p d')  # frame, head, pixel_num, dim -> head, frame, pixel_dim
        # repeating
        k = k.repeat(frame_num, 1, 1, 1)  # k = [1, headm pixel_num, dim] -> frame, head
        k = rearrange(k, 'f h p d -> h f p d')  # 1, head, pixel_num, dim -> head, f*pixel_num, dim
        v = v.repeat(frame_num, 1, 1, 1)
        v = rearrange(v, 'f h p d -> h f p d')  # 1, head, pixel_num, dim -> head, f*pixel_num, dim
        # [1]
        dropout_p = 0.0
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
        attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [head, frame*pixel, pixel]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)  # [batch*pixel_num,head, frame, frame] # last frame is only one
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        hidden = attn_weight @ v  # head, (2*pixel_num) dim
        hidden = rearrange(hidden, 'h f p d -> f h p d', f=frame_num)  # 2, 8, pixel_num, dim
        hidden = hidden.transpose(1, 2)  # [frame, pixel_num, head_num, dim]
        head_num, head_dim = hidden.shape[-2], hidden.shape[-1]
        hidden = hidden.reshape(batch_size, -1, head_num * head_dim)  # [batch*frame, pixel_num, dim]
        return hidden

    def masactrl_forward(self, q, k, v, head_num, head_dim, batch_size):

        # q = [batch*frame, head, pixel_num, dim]
        qu, qc = q.chunk(2, dim=0)  # [2, head, pixel, dim] [2, head, pixel, dim]
        ku, kc = k.chunk(2, dim=0)  # [frame, head, pixel, dim]
        vu, vc = v.chunk(2, dim=0)
        ku, kc = ku[0, :, :, :].unsqueeze(0), kc[0, :, :, :].unsqueeze(0)  # [1,head, pixel, dim]
        vu, vc = vu[0, :, :, :].unsqueeze(0), vc[0, :, :, :].unsqueeze(0)

        out_u = self.masactrl_attention(qu, ku, vu)
        out_c = self.masactrl_attention(qc, kc, vc)
        out = torch.cat([out_u, out_c], dim=0)  # [frame*batch, head, pixel dim]

        return out

    ##
    def repeat_attention_cross(self, q, k, v, head_num, head_dim, batch_size, layer_name):
        # print(f'cross attention {layer_name} repeat')
        kc, ku = torch.chunk(k, chunks=2, dim=0)
        # vc, vu = torch.chunk(v, chunks=2, dim=0)
        k = torch.cat([kc, ku], dim=0)
        # v = torch.cat([vc, vu], dim=0)
        # [3]
        dropout_p = 0.0
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
        attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [batch*pixel_num/head/frame/frame]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight,
                                    dim=-1)  # [batch*pixel_num, head, frame, frame] # last frame is only one
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        # [2]
        attn_c, attn_u = torch.chunk(attn_weight, 2, dim=0)

        f_attn_c = attn_c[0, :, :, :].unsqueeze(0)
        f_attn_u = attn_u[0, :, :, :].unsqueeze(0)
        attn_c[1:, :, :, :] = f_attn_c.repeat(self.frame_num - 1, 1, 1, 1)
        attn_u[1:, :, :, :] = f_attn_u.repeat(self.frame_num - 1, 1, 1, 1)
        """
        subject_attn_c = attn_c[:, :, :, 2].unsqueeze(0)
        subject_attn_u = attn_u[:, :, :, 2].unsqueeze(0)
        attn_c[:, :, :, 3] = subject_attn_c
        attn_u[:, :, :, 3] = subject_attn_u        
        """
        attn_weight = torch.cat([attn_c, attn_u], dim=0)  # [batch*pixel_num, head, frame, sen_]

        if self.do_save_attention_map:
            # print(f'saving! type of attn_weight = {type(attn_weight)}')
            # if layer_name not in self.attnmap_dict :
            #    self.attnmap_dict[layer_name] = []
            # self.attnmap_dict[layer_name].append(attn_weight)
            self.attnmap_dict[layer_name] = attn_weight

        # [3]
        hidden_states = attn_weight @ v  # [pixel_num, head, frame, dim]
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, head_num * head_dim)  # [batch*pixel_num, frame, dim]
        hidden_states = hidden_states.to(q.dtype)
        return hidden_states

    def base_forward(self, q, k, v, head_num, head_dim, batch_size, layer_name):
        dropout_p = 0.0
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
        attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [batch*pixel_num/head/frame/frame]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight,  dim=-1)  # [batch*pixel_num,head, frame, frame] # last frame is only one
        hidden_states = attn_weight @ v  # [pixel_num, head, frame, dim]
        hidden_states = hidden_states.transpose(1, 2) # pixel_num, frame, head, dim
        hidden_states = hidden_states.reshape(batch_size, -1, head_num * head_dim)  # [batch*pixel_num, frame, dim]
        return hidden_states

    def base_forward_cross(self, q, k, v, head_num, head_dim, batch_size, layer_name):

        # q = [batch*frame, head, pixel_num, dim]
        pixel_num = q.shape[-2]
        res = int(pixel_num ** 0.5)  # 64
        half_res = int(res // 2)

        # query = [[1, 1, 1, 1],
        #          [1, 1, 1, 1],
        #          [1, 1, 1, 1],
        #          [1, 1, 1, 1],]
        # get only first half of the query
        # target_query = [[1, 1],
        #                 [1, 1]]

        batch_size, head, pixel_num, dim = q.shape
        rest_q = q
        q = q.reshape(batch_size, head, res, res, dim)
        trg_q    = q[:, :, :half_res, :half_res, :]
        second_q = q[:, :, :half_res, half_res:, :]
        third_q = q[:, :, half_res:, half_res:, :]
        forth_q = q[:, :, half_res:, :half_res, :]

        first_q = trg_q.reshape(batch_size, head, half_res * half_res, dim)
        second_q = second_q.reshape(batch_size, head, half_res * half_res, dim)
        third_q = third_q.reshape(batch_size, head, half_res * half_res, dim)
        forth_q = forth_q.reshape(batch_size, head, half_res * half_res, dim)

        total_len = k.size(-2)

        first_k = k[:, :, :int(total_len / 2), :]
        rest_k = k[:, :, int(total_len / 2):, :]
        first_v = v[:, :, :int(total_len / 2), :]
        rest_v = v[:, :, int(total_len / 2):, :]

        # [1] first q attention
        def do_attn(q, k, v):
            dropout_p = 0.0
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)
            attn_weight = q @ k.transpose(-2, -1) * scale_factor  # [batch*pixel_num/head/frame/frame]
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight,
                                        dim=-1)  # [batch*pixel_num,head, frame, frame] # last frame is only one
            hidden_states = attn_weight @ v  # [pixel_num, head, frame, dim]
            return hidden_states

        first_q = do_attn(first_q, first_k, first_v)
        first_q = first_q.reshape(batch_size, head, half_res, half_res, dim)

        second_q = do_attn(second_q,first_k, first_v)
        second_q = second_q.reshape(batch_size, head, half_res, half_res, dim)
        third_q = do_attn(third_q, rest_k, rest_v)
        third_q = third_q.reshape(batch_size, head, half_res, half_res, dim)
        forth_q = do_attn(forth_q, rest_k, rest_v)
        forth_q = forth_q.reshape(batch_size, head, half_res, half_res, dim)
        q[:, :, :half_res, :half_res, :] = first_q
        q[:, :, :half_res, half_res:, :] = second_q
        q[:, :, half_res:, half_res:, :] = third_q
        q[:, :, half_res:, :half_res, :] = forth_q
        hidden_states = q.reshape(batch_size, head, res * res, dim)

        # hidden_states = rest_q

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, head_num * head_dim)  # [batch*pixel_num, frame, dim]
        return hidden_states
    
    def temporal_window_attention(self, q,k,v):
        
        dropout_p = 0.0
        L, S = q.size(-2), k.size(-2)
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_bias = torch.zeros(L, S, dtype=q.dtype).to(q.device)                   
        
        if self.temporal_firstframe_attention :
            k = k[:,:,0,:].unsqueeze(2)
            v = v[:,:,0,:].unsqueeze(2)
        
        attn_weight = q @ k.transpose(-2,-1) * scale_factor  # [batch*pixel_num/head/frame/dim] [batch*pixel_num/head/dim/frame]
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)  # [batch*pixel_num,head, frame, frame]
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        hidden_states = attn_weight @ v  # [pixel_num, head, frame, dim]
        return hidden_states

    def reset(self):

        # [2] state
        self.attnmap_dict = {}
        self.timestep = 0
        self.layerwise_hidden_dict = {}
        self.spatial_iter = 0
        self.cross_iter = 0
        self.time_idx = -1
        self.cached_motion_states = {}