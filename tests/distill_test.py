import torch, gc
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
import argparse, json
from prun.utils import arg_as_list
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput, TransformerTemporalModel
import torch.nn as nn
from typing import Any, Dict, Optional
from diffusers import (AutoencoderKL, DDPMScheduler, LCMScheduler, StableDiffusionPipeline, UNetMotionModel)
import numpy as np
from pathlib import Path
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
import wandb, csv
from copy import deepcopy
import sys
from prun.data import DistillWebVid10M
from torch.utils.data import RandomSampler
from diffusers.optimization import get_scheduler
from einops import rearrange
import math, os
import functools
import random
import torch.nn.functional as F
from tqdm.auto import tqdm
import logging
from diffusers.utils import export_to_gif, export_to_video
import GPUtil
from diffusers.utils import check_min_version
from prun.utils.block_info import total_blocks, total_blocks_dot


class SimpleAttention(nn.Module):

    def __init__(self, dim, heads=8, layer_name=""):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        # self.to_out = nn.Linear(dim, dim)
        # non linear layer
        #
        self._zero_initialize()
        self.layer_name = layer_name

    def _zero_initialize(self):
        self.to_qkv.weight.data.zero_()
        self.to_qkv.bias.data.zero_()
        # self.to_out.weight.data.zero_()
        # self.to_out.bias.data.zero_()

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

        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=-1)  # [batch*pixel, num_frames, dim]
        # multihead attention
        head_dim = channel // self.heads
        query = query.view(-1, num_frames, self.heads, head_dim).transpose(1,
                                                                           2)  # [batch*pixel, head, num_frames, head_dim]
        key = key.view(-1, num_frames, self.heads, head_dim).transpose(1,
                                                                       2)  # [batch*pixel, head, num_frames, head_dim]
        value = value.view(-1, num_frames, self.heads, head_dim).transpose(1, 2)  # [batch*pixel, head, num_frames, head_dim]

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
        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


def set_simple_attention(unet,
                         skip_layers : list):

    def register_editor(net_name, net, skip_layers) :

        for name, subnet in net.named_children():
            final_name = f"{net_name}_{name}"
            if subnet.__class__.__name__ == 'TransformerTemporalModel' or subnet.__class__.__name__ == 'AnimateDiffTransformer3D':
                if final_name in skip_layers :
                    print(f'final_name = {final_name} in skip_layers')
                    basic_dim = subnet.proj_in.in_features
                    print(f'setting simple attention')
                    simple_block = SimpleAttention(basic_dim, layer_name=final_name)
                    setattr(net, name, simple_block)
                    subnet = simple_block
            if hasattr(net, 'children'):
                register_editor(final_name, subnet, skip_layers)

    for net_name, net in unet.named_children():
        register_editor(net_name, net, skip_layers)



def evaluation(prompt, evaluation_pipe, controller, save_folder,
               n_prompt, num_frames, guidance_scale, num_inference_steps, h=512, weight_dtype=torch.float16, p=0):
    prompt = prompt.strip()
    with torch.no_grad() :
        output = evaluation_pipe(prompt=prompt,
                                 negative_prompt=n_prompt,
                                 num_frames=num_frames,
                                 guidance_scale=guidance_scale,
                                 num_inference_steps=num_inference_steps,
                                 height=h,
                                 width=h,
                                 generator=torch.Generator("cpu").manual_seed(args.seed))
    if controller is not None :
        controller.reset()
        frames = output.frames[0]
        export_to_video(frames, os.path.join(save_folder, f'pruned_{prompt}.mp4'))
    else :
        frames = output.frames[0]

        #export_to_video(frames, os.path.join(save_folder, f'teacher_{prompt}.mp4'))
        export_to_video(frames, os.path.join(save_folder, f'sample_{p}.mp4'))
    # save on txt file
    text_file = os.path.join(save_folder, 'sample.txt')
    with open(text_file, 'a') as f :
        f.write(f'sample_{p},sample_{p}.mp4,{prompt}\n')

def main(args):

    torch.distributed.init_process_group(backend='nccl')

    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    print(f' step 1. logger and seed')
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_config=accelerator_project_config,
                              split_batches=True, )
    is_main_process = accelerator.is_local_main_process

    logger.info(f'\n step 2. preparing folder')
    logger.info(f' (2.1) seed')
    if args.seed is not None:
        set_seed(args.seed)
    logger.info(f' (2.2) saving dir')
    output_dir = args.output_dir
    sample_dir = os.path.join(output_dir, "student_model_sample")
    os.makedirs(sample_dir, exist_ok=True)
    epoch_dir = os.path.join(sample_dir, args.trg_epoch) # save dir
    os.makedirs(epoch_dir, exist_ok=True)


    print(f'\n step 2. noise scheduler and solver')
    weight_dtype = torch.float32
    student_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    student_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=student_adapter,torch_dtype=weight_dtype)
    student_pipe.scheduler = LCMScheduler.from_config(student_pipe.scheduler.config, beta_schedule="linear")
    student_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    student_pipe.set_adapters(["lcm-lora"], [0.8])
    student_pipe.enable_vae_slicing()
    student_pipe.enable_model_cpu_offload()
    total_block_num = 21
    teacher_architecture = [i for i in range(total_block_num)]
    using_block_num = len(args.architecture)
    prun_arch = [x for x in teacher_architecture if x not in args.architecture]
    skip_layers = [total_blocks[i] for i in prun_arch]
    print(f'skip_layers : {skip_layers}')
    skip_layers_dot = [total_blocks_dot[i] for i in prun_arch]
    student_unet = student_pipe.unet
    set_simple_attention(student_unet, skip_layers)

    # register hook
    print(f'\n step 3. load')
    if args.pretrained is not None:
        student_unet.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
        student_unet = student_unet.to(dtype = weight_dtype)
        student_unet = student_unet.to(accelerator.device)
    student_pipe.unet = student_unet

    print(f'\n step 4. inference')
    logger.info(f'\n step 8. inference with prun model')
    custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        custom_prompts = f.readlines()
    n_prompt = "bad quality, worse quality, low resolution"
    num_frames = args.sample_n_frames
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_step
    weight_dtype = torch.float32
    for p, prompt in enumerate(custom_prompts):
        if p < 50:
            evaluation(prompt,
                       student_pipe,
                       None,
                       epoch_dir,
                       n_prompt,
                       num_frames,
                       guidance_scale,
                       num_inference_steps,
                       h=512,
                       weight_dtype=weight_dtype,
                       p=p)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # [1]
    parser.add_argument("--output_dir", type=str, default="emilianJR/epiCRealism")
    parser.add_argument("--logging_dir", type=str, default="emilianJR/epiCRealism")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_run_name", type=str, default="emilianJR/epiCRealism")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."))
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")

    parser.add_argument("--pretrained_teacher_model", type=str, default="emilianJR/epiCRealism",
                        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.", )

    parser.add_argument("--pretrained", type=str, default=None, required=False)
    parser.add_argument("--num_ddim_timesteps", type=int, default=50,
                        help="The number of timesteps to use for DDIM sampling.", )
    # [2]
    # using arch = [0, 1, 2, 3, 4, 6, 14, 15, 16, 20]
    parser.add_argument("--architecture", type=arg_as_list,
                        default=[0, 1, 2, 3, 4, 6, 14, 15, 16, 20])
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"), )
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.", )
    # [7]
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # [9]
    parser.add_argument("--datavideo_size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--num_inference_step", type=int, default=6)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--per_gpu_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--video_folder", type=str)
    # [9]
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_train_steps", type=int, default=50000,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.", )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'))
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.", )
    parser.add_argument("--scale_lr", action="store_true", default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.", )
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "huber"],
                        help="The type of loss to use for the LCD loss.", )
    parser.add_argument("--huber_c", type=float, default=0.001,
                        help="The huber loss parameter. Only used if `--loss_type=huber`.", )
    parser.add_argument("--feature_matching_loss_weight", type=float, default=1.0)
    parser.add_argument("--trg_epoch", type = str)

    args = parser.parse_args()
    main(args)
