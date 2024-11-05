import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, Optional
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput, TransformerTemporalModel
from .controller import AttentionBase
import math
import torch.nn.functional as F
import os
import numpy as np
import torch
from .sparsify import sparsify_layer, pruned_output



def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )
    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim, )
    return ff_output


class SinusoidalPositionalEmbedding_custom(nn.Module):

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x





def register_motion_editor(unet, editor: AttentionBase):

    class SimpleAttention(nn.Module):

        def __init__(self, dim, heads=8, layer_name=""):
            super().__init__()
            self.heads = heads
            self.to_qkv = nn.Linear(dim, dim * 3)
            #self.to_out = nn.Linear(dim, dim)
            self._zero_initialize()
            self.layer_name = layer_name

        def _zero_initialize(self):
            self.to_qkv.weight.data.zero_()
            self.to_qkv.bias.data.zero_()
            #self.to_out.weight.data.zero_()
            #self.to_out.bias.data.zero_()

        def forward(self,
                    hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.LongTensor] = None,
                    timestep: Optional[torch.LongTensor] = None,
                    class_labels: torch.LongTensor = None,
                    num_frames: int = 1,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    return_dict: bool = True, ) -> TransformerTemporalModelOutput:
            # [1] reshaping
            batch_frames, channel, height, width = hidden_states.shape  # batch_frames = frame
            batch_size = batch_frames // num_frames
            residual = hidden_states
            hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # [batch, dim, frame, height, width]
            height = hidden_states.shape[3]

            hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames,
                                                                         channel)
            query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)  # [batch*pixel, num_frames, dim]
            # multihead attention
            head_dim = channel // self.heads
            query = query.view(-1, num_frames, self.heads, head_dim).transpose(1,
                                                                               2)  # [batch*pixel, head, num_frames, head_dim]
            key = key.view(-1, num_frames, self.heads, head_dim).transpose(1,
                                                                           2)  # [batch*pixel, head, num_frames, head_dim]
            value = value.view(-1, num_frames, self.heads, head_dim).transpose(1,
                                                                               2)  # [batch*pixel, head, num_frames, head_dim]

            # [3] attention
            attn = (query @ key.transpose(-2, -1))  # [batch*pixel, head, num_frames, num_frames]
            attn = attn.softmax(dim=-1)
            out = attn @ value  # [batch*pixel, head, num_frames, head_dim]
            out = out.transpose(1, 2)  # [batch*pixel, num_frames, head, head_dim]
            out = out.reshape(batch_size, height, width, num_frames,
                              channel)  # [batch, height, width, num_frames, channel]
            out = out.permute(0, 3, 4, 1, 2).contiguous()  # [batch, num_frames, channel, height, width]
            out = out.reshape(batch_frames, channel, height, width)  # [batch_frame, channel, height, width]
            output = out + residual
            editor.save_hidden_states(output, self.layer_name)
            if not return_dict:
                return (output,)
            return TransformerTemporalModelOutput(sample=output)



    def motion_forward_basic(self, layer_name):

        def forward(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.LongTensor] = None,
                    timestep: Optional[torch.LongTensor] = None,
                    class_labels: torch.LongTensor = None,
                    num_frames: int = 1,
                    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                    return_dict: bool = True, ) -> TransformerTemporalModelOutput:

            do_skip = False
            #for skip_layer in editor.skip_layers:
            #    if skip_layer == layer_name.lower() and editor.timestep == editor.target_time:
            #        do_skip = True

            #high_res = False

            if do_skip:
                residual = hidden_states  # [batch_frame, channel, height, width]
                output = residual
                if not return_dict:
                    return (output,)
                return TransformerTemporalModelOutput(sample=output)

            else:
                batch_frames, channel, height, width = hidden_states.shape  # batch_frames = frame
                batch_size = batch_frames // num_frames
                residual = hidden_states
                hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)  # [batch, dim, frame, height, width]
                hidden_states = self.norm(hidden_states)  # 2, dim, frame, height, width
                height = hidden_states.shape[3]
                hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)
                hidden_states = self.proj_in(hidden_states)

                for block in self.transformer_blocks:
                    hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states,
                                          timestep=timestep, cross_attention_kwargs=cross_attention_kwargs,
                                          class_labels=class_labels, )
                # [2] linear layer
                hidden_states = self.proj_out(hidden_states)  # batch*h*w, num_frame, channel
                hidden_states = (hidden_states[None, None, :].reshape(batch_size, height, width, num_frames, channel).permute(0,3,4,1,2).contiguous())
                hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
                output = hidden_states + residual
                editor.save_hidden_states(output, layer_name)

            if not return_dict:
                return (output,)
            return TransformerTemporalModelOutput(sample=output)
        return forward

    def motion_feedforward(self, layer_name):

        def forward(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            for module in self.net:
                hidden_states = module(hidden_states)
            return hidden_states
        return forward

    def motion_forward_basictransformerblock(self, layer_name):

        def forward(hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                timestep: Optional[torch.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[torch.LongTensor] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,) -> torch.Tensor:

            control = False
            for skip_layer_name in editor.skip_layers:
                if skip_layer_name in layer_name:
                    control = True

            high_res = False
            # if "down_blocks_0_motion_modules" in layer_name.lower() or 'up_blocks_3_motion_modules' in layer_name.lower():
            #    high_res = True

            if high_res and editor.window_attention:
                hidden_states_list = hidden_states
                batch_size = hidden_states_list[0].shape[0]
                if self.norm_type == "ada_norm":
                    norm_hidden_states_list = [self.norm1(hidden_states, timestep) for hidden_states in hidden_states_list]
                elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                    chunk_size = len(hidden_states_list)  # one dim norm (normalizing in embedding dim)
                    norm_hidden_states = self.norm1(
                        torch.cat(hidden_states_list, dim=0))  # [batch*pixel_num, frame, dim]
                    norm_hidden_states_list = torch.chunk(norm_hidden_states, chunk_size, dim=0)
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states_list = [self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"]) for hidden_states in hidden_states_list]
                else:
                    raise ValueError("Incorrect norm used")
                if self.pos_embed is not None:
                    norm_hidden_states_list = [self.pos_embed(norm_hidden_states) for norm_hidden_states in norm_hidden_states_list]
                # 1. Prepare GLIGEN inputs
                cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
                gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
                attn_output_list = self.attn1(norm_hidden_states_list,
                                              encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                              attention_mask=attention_mask,
                                              **cross_attention_kwargs, )
                if self.norm_type == "ada_norm_zero":
                    attn_output_list = [gate_msa.unsqueeze(1) * attn_output for attn_output in attn_output_list]
                elif self.norm_type == "ada_norm_single":
                    attn_output_list = [gate_msa * attn_output for attn_output in attn_output_list]
                hidden_states_list = [attn_output + hidden_states for attn_output, hidden_states in
                                      zip(attn_output_list, hidden_states_list)]
                if hidden_states_list[0].ndim == 4:
                    hidden_states_list = [hidden_states.squeeze(1) for hidden_states in hidden_states_list]
                # 1.2 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states_list = [self.fuser(hidden_states, gligen_kwargs["objs"]) for hidden_states in
                                          hidden_states_list]
                # 3. Cross-Attention
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states_list = [self.norm2(hidden_states, timestep) for hidden_states in
                                                   hidden_states_list]
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        chunk_size = len(hidden_states_list)
                        norm_hidden_states = self.norm2(torch.cat(hidden_states_list, dim=0))
                        norm_hidden_states_list = torch.chunk(norm_hidden_states, chunk_size, dim=0)
                    elif self.norm_type == "ada_norm_single":
                        norm_hidden_states_list = hidden_states_list
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states_list = [self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"]) for
                                                   hidden_states in hidden_states_list]
                    else:
                        raise ValueError("Incorrect norm")
                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states_list = [self.pos_embed(norm_hidden_states) for norm_hidden_states in
                                                   norm_hidden_states_list]
                    attn_output_list = self.attn2(norm_hidden_states_list,
                                                  encoder_hidden_states=encoder_hidden_states,
                                                  attention_mask=encoder_attention_mask,
                                                  **cross_attention_kwargs, )
                    hidden_states_list = [attn_output + hidden_states for attn_output, hidden_states in
                                          zip(attn_output_list, hidden_states_list)]
                # 4. Feed-forward
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states_list = [self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"]) for
                                               hidden_states in hidden_states_list]
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states_list = [self.norm3(hidden_states) for hidden_states in hidden_states_list]
                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states_list = [norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None] for
                                               norm_hidden_states in norm_hidden_states_list]
                if self.norm_type == "ada_norm_single":
                    norm_hidden_states_list = [self.norm2(hidden_states) for hidden_states in hidden_states_list]
                    norm_hidden_states_list = [norm_hidden_states * (1 + scale_mlp) + shift_mlp for norm_hidden_states
                                               in norm_hidden_states_list]
                if self._chunk_size is not None:
                    ff_output_list = [
                        _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size) for
                        norm_hidden_states in norm_hidden_states_list]
                else:
                    ff_output_list = [self.ff(norm_hidden_states) for norm_hidden_states in norm_hidden_states_list]
                if self.norm_type == "ada_norm_zero":
                    ff_output_list = [gate_mlp.unsqueeze(1) * ff_output for ff_output in ff_output_list]
                elif self.norm_type == "ada_norm_single":
                    ff_output_list = [gate_mlp * ff_output for ff_output in ff_output_list]
                # block scaling checking
                hidden_states_list = [ff_output + hidden_states for ff_output, hidden_states in
                                      zip(ff_output_list, hidden_states_list)]
                if hidden_states_list[0].ndim == 4:
                    hidden_states_list = [hidden_states.squeeze(1) for hidden_states in hidden_states_list]
                return hidden_states_list

            else:
                if cross_attention_kwargs is not None:
                    if cross_attention_kwargs.get("scale", None) is not None:
                        logger.warning(
                            "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
                batch_size = hidden_states.shape[0]
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm1(hidden_states, timestep)
                elif self.norm_type == "ada_norm_zero":
                    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                        hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                    )
                elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm1(hidden_states)  # ndim = 3
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif self.norm_type == "ada_norm_single":
                    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)).chunk(6, dim=1)
                    norm_hidden_states = self.norm1(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                else:
                    raise ValueError("Incorrect norm used")

                if self.pos_embed is not None:
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                # 1. Prepare GLIGEN inputs
                cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
                gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
                attn_output = self.attn1(norm_hidden_states,
                                         encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                                         attention_mask=attention_mask, **cross_attention_kwargs, )
                if self.norm_type == "ada_norm_zero":
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.norm_type == "ada_norm_single":
                    attn_output = gate_msa * attn_output
                hidden_states = attn_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

                # 1.2 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

                # 3. Cross-Attention
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        hidden_states = self.norm2(hidden_states)  # ndim = 3
                    elif self.norm_type == "ada_norm_single":
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                    else:
                        raise ValueError("Incorrect norm")
                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=encoder_attention_mask, **cross_attention_kwargs, )
                    hidden_states = attn_output + hidden_states

                # 4. Feed-forward
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)
                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                if self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
                if self._chunk_size is not None:
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    # ff layer
                    ff_output = self.ff(norm_hidden_states)
                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output
                hidden_states = ff_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)
                return hidden_states

        return forward

    def motion_forward(self, layer_name):

        def forward(hidden_states: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs, ) -> torch.Tensor:
            high_res = False
            # if "down_blocks_0_motion_modules" in full_name.lower() or 'up_blocks_3_motion_modules' in full_name.lower():
            #    high_res = True

            control = False
            for skip_layer in editor.skip_layers:
                if skip_layer in layer_name:
                    control = True
                   #mask_model = editor.mask_dict[skip_layer]

            if high_res and editor.window_attention:
                attn = self
                residual_list = hidden_states  # [batch*pixel_num, frame, dim]
                hidden_states_list = hidden_states
                if attn.spatial_norm is not None:
                    hidden_states_list = [attn.spatial_norm(hidden_states, temb) for hidden_states in
                                          hidden_states_list]
                input_ndim = hidden_states_list[0].ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1,
                                                                                                      2)  # batch*frame, pixel_num, dim
                batch_size, sequence_length, _ = (
                    hidden_states_list[0].shape if encoder_hidden_states is None else encoder_hidden_states.shape)
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                query_list = [attn.to_q(hidden_states) for hidden_states in
                              hidden_states_list]  # [batch*pixel_num, frame=16, dim]
                # [2] key and value
                if encoder_hidden_states is None:
                    encoder_hidden_states_list = hidden_states_list
                elif attn.norm_cross:
                    encoder_hidden_states_list = [attn.norm_encoder_hidden_states(encoder_hidden_states) for
                                                  encoder_hidden_states in encoder_hidden_states_list]
                key_list = [attn.to_k(encoder_hidden_states) for encoder_hidden_states in
                            encoder_hidden_states_list]  # [batch*pixel_num, frame_num, dim]
                value_list = [attn.to_v(encoder_hidden_states) for encoder_hidden_states in
                              encoder_hidden_states_list]  # [batch*pixel_num, frame_num, dim]
                inner_dim = key_list[0].shape[-1]
                head_dim = inner_dim // attn.heads
                query_list = [query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for query in
                              query_list]  # [pixel_num, frame, head, head_dim] -> [batch*pixel_num/head/frame/dim]
                key_list = [key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for key in
                            key_list]  # [pixel_num, head, frame, dim] # frame wise attention
                value_list = [value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for value in
                              value_list]  # [pixel_num, head, frame, dim] # frame wise attention
                out_list = []
                for query, key, value in zip(query_list, key_list, value_list):
                    hidden_states = editor.temporal_window_attention(query, key, value)
                    out_list.append(hidden_states)
                hidden_states_list = [hidden_states.transpose(1, 2) for hidden_states in out_list]
                hidden_states_list = [hidden_states.reshape(batch_size, -1, attn.heads * head_dim) for hidden_states in
                                      hidden_states_list]
                hidden_states_list = [hidden_states.to(query.dtype) for hidden_states in hidden_states_list]
                hidden_states_list = [attn.to_out[0](hidden_states) for hidden_states in hidden_states_list]
                hidden_states_list = [attn.to_out[1](hidden_states) for hidden_states in hidden_states_list]
                if input_ndim == 4:
                    hidden_states_list = [hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                                          for hidden_states in hidden_states_list]
                if attn.residual_connection:
                    hidden_states_list = [hidden_states + residual for hidden_states, residual in
                                          zip(hidden_states_list, residual_list)]
                hidden_states_list = [hidden_states / attn.rescale_output_factor for hidden_states in
                                      hidden_states_list]
                return hidden_states_list

            else:
                attn = self
                residual = hidden_states  # [batch*pixel_num, frame, dim]
                high_res = False
                if "down_blocks_0_motion_modules" in layer_name.lower() or 'up_blocks_3_motion_modules' in layer_name.lower():
                    high_res = True
                if attn.spatial_norm is not None:
                    hidden_states = attn.spatial_norm(hidden_states, temb)
                input_ndim = hidden_states.ndim
                if input_ndim == 4:
                    batch_size, channel, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)  # batch*frame, pixel_num, dim
                batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
                if attention_mask is not None:
                    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # ---------------------------------------------------------------------------------------------------------------------------------- #
                query = attn.to_q(hidden_states)  # [batch*pixel_num, frame=16, dim] (dimension change_)

                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states

                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states)  # [batch*pixel_num, frame=16, dim] (dimension change_)
                value = attn.to_v(encoder_hidden_states)  # [batch*pixel_num, frame=16, dim] (dimension change_)
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads

                # ===================================================================================================== #
                # if control and editor.remove_attn_head :
                #    query =

                # else :
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                dropout_p = 0.0
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
                attn_weight = query @ key.transpose(-2, -1) * scale_factor  # [batch*pixel_num/head/frame/dim] [batch*pixel_num/head/dim/frame]
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)  # [batch*pixel_num,head, frame, frame]
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                hidden_states = attn_weight @ value  # [pixel_num, head, frame, dim]
                hidden_states = hidden_states.transpose(1, 2)  # batch_size, frame_num, head, dim     # [1] concat
                hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)  # [batch*pixel_num, frame, dim]
                # if i erase special head, output of hidden_states shorter
                # batch_size, -1, shorted dimension
                # to_out[0] input dim should be shortened ...
                hidden_states = hidden_states.to(query.dtype)
                # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                # fead forward
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states

        return forward

    def attention_forward(self, layer_name):

        def forward(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs, ) -> torch.Tensor:
            is_cross_atten = True
            if encoder_hidden_states is None:
                is_cross_atten = False
                editor.spatial_iter += 1
            else:
                # print(f'cross attention layer name = {layer_name}')
                editor.timestepwise()
                editor.cross_iter += 1

            crossattn_control = False
            attn = self
            residual = hidden_states
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # frame, head, pixel_num, dim
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                return attn_weight @ value

            def scaled_dot_product_attention_cross(query, key, value, dropout_p=0.0) -> torch.Tensor:
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1))
                attn_bias = torch.zeros(L, S, dtype=query.dtype).to(query.device)
                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                timestep = editor.timestep
                # editor.save_hidden_states(attn_weight, layer_name)
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                return attn_weight @ value

            if is_cross_atten:
                hidden_states = scaled_dot_product_attention_cross(query, key, value)
            else:
                hidden_states = scaled_dot_product_attention(query, key, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if attn.residual_connection:
                hidden_states = hidden_states + residual
            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        return forward
    """
    def forward_hook(module, input, output, storage):
        storage.append(output.detach())  # detach()를 통해 그래디언트와의 연결을 끊음

    def register_editor(net, net_name, output_storage):
        # forward hook을 등록하여 모델의 출력을 저장
        hook = net.register_forward_hook(lambda m, i, o: forward_hook(m, i, o, output_storage))
        return hook
    """
    def register_editor(net, count, place_in_unet, net_name):

        for name, subnet in net.named_children():
            final_name = f"{net_name}_{name}"

            # train only this module

            if subnet.__class__.__name__ == 'TransformerTemporalModel' or subnet.__class__.__name__ == 'AnimateDiffTransformer3D':
                if final_name in editor.skip_layers:

                    if not editor.is_teacher :
                        basic_dim = subnet.proj_in.in_features
                        simple_block = SimpleAttention(basic_dim, layer_name=final_name)
                        setattr(net, name, simple_block)
                        subnet = simple_block

                    else:
                        subnet.forward = motion_forward_basic(subnet, final_name)


                # caching the output (only the

            #if subnet.__class__.__name__ == 'BasicTransformerBlock' and 'motion' in final_name.lower():
            #    subnet.forward = motion_forward_basictransformerblock(subnet, final_name)

            #if subnet.__class__.__name__ == 'FeedForward' and 'motion' in final_name.lower():
            #    subnet.forward = motion_feedforward(subnet, final_name)

            # if subnet.__class__.__name__ == 'Attention' and 'motion' not in final_name.lower():
            #    subnet.forward = attention_forward(subnet, final_name)

            if subnet.__class__.__name__ == 'Attention' and 'motion' in final_name.lower():
                subnet.forward = motion_forward(subnet, final_name)  # attention again

            if hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet, final_name)

        return count

    cross_att_count = 0
    for net_name, net in unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down", net_name)
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid", net_name)
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up", net_name)
    editor.num_att_layers = cross_att_count


