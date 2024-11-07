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
class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

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

def main(args) :

    print(f' step 0. distribution setting')
    torch.distributed.init_process_group(backend='nccl')
    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    print(f' step 1. logger and seed')
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,  log_with=args.report_to,
                              project_config=accelerator_project_config, split_batches=True, )
    is_main_process = accelerator.is_main_process
    if is_main_process:
        run = wandb.init(project='ondevice-video', name=args.wandb_run_name, group="DDP", )

    logger.info(f'\n step 2. preparing folder')
    logger.info(f' (2.1) seed')
    if args.seed is not None:
        set_seed(args.seed)
    logger.info(f' (2.2) saving dir')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f' (2.3) logging dir')

    print(f' step 3. weight and device')
    weight_dtype = torch.float32  # use torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":  # here
        weight_dtype = torch.bfloat16

    print(f'\n step 4. noise scheduler and solver')
    dir = '/home/dreamyou070/.cache/huggingface/hub/models--emilianJR--epiCRealism/snapshots/6522cf856b8c8e14638a0aaa7bd89b1b098aed17'
    noise_scheduler = DDPMScheduler.from_pretrained(dir,
                                                    subfolder="scheduler")
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), timesteps=noise_scheduler.config.num_train_timesteps, ddim_timesteps=args.num_ddim_timesteps, )

    print(f'\n step 5. model and pipe')
    print(f' (1) teacher pipe')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(dir, motion_adapter=teacher_adapter, torch_dtype=weight_dtype)
    teacher_pipe.scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])
    teacher_pipe.enable_vae_slicing()
    teacher_pipe.enable_model_cpu_offload()
    teacher_unet = teacher_pipe.unet


    print(f' (2) student pipe')
    student_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    student_pipe = AnimateDiffPipeline.from_pretrained(dir, motion_adapter=student_adapter,torch_dtype=weight_dtype)
    student_pipe.scheduler = LCMScheduler.from_config(student_pipe.scheduler.config, beta_schedule="linear")
    student_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    student_pipe.set_adapters(["lcm-lora"], [0.8])
    student_pipe.enable_vae_slicing()
    student_pipe.enable_model_cpu_offload()
    total_block_num = 21
    teacher_architecture = [i for i in range(total_block_num)]
    using_block_num = len(args.architecture)
    prun_arch = [x for x in teacher_architecture if x not in args.architecture]
    skip_layers = [total_blocks[i] for i in prun_arch]
    skip_layers_dot = [total_blocks_dot[i] for i in prun_arch]
    student_unet = student_pipe.unet
    set_simple_attention(student_unet, skip_layers)
    student_unet.to(dtype=weight_dtype)

    # register hook
    def register_hook(unet, hook_fn):
        hooked_subnets = set()  # 후크가 등록된 서브넷을 추적하는 집합
        def hooking(net):
            for name, subnet in net.named_children():
                # 서브넷이 후크가 이미 등록된 경우 무시
                if subnet in hooked_subnets:
                    continue
                # 후크를 등록할 조건
                if subnet.__class__.__name__ == 'TransformerTemporalModel' or subnet.__class__.__name__ == 'AnimateDiffTransformer3D':
                    subnet.register_forward_hook(hook_fn)
                    hooked_subnets.add(subnet)  # 후크 등록 후 추가
                if subnet.__class__.__name__ == 'SimpleAttention':
                    subnet.register_forward_hook(hook_fn)
                    hooked_subnets.add(subnet)
                if hasattr(subnet, 'children'):
                    hooking(subnet)
        hooking(unet)

    def hook_fn_teacher(module, input, output):
        if type(input) != torch.Tensor:
            input = input[0]
        if type(output) != torch.Tensor:
            output = output[0]
        inputs_teacher.append(input.detach())  # Teacher 모델의 입력
        outputs_teacher.append(output.detach())  # Teacher 모델의 출력
    register_hook(teacher_unet, hook_fn_teacher)

    def hook_fn_student(module, input, output):
        if type(input) != torch.Tensor:
            input = input[0]
        if type(output) != torch.Tensor:
            output = output[0]
        inputs_student.append(input.detach())  # Student 모델의 입력
        outputs_student.append(output.detach())  # Student 모델의 출력
    register_hook(student_unet, hook_fn_student)

    print(f' (3) sub models')
    device = torch.device("cuda")
    vae = teacher_pipe.vae
    vae = vae.to(device=device, dtype=weight_dtype)
    text_encoder = teacher_pipe.text_encoder
    text_encoder = text_encoder.to(device=device, dtype=weight_dtype)
    tokenizer = teacher_pipe.tokenizer
    solver = solver.to(device=device)

    print(f'\n step 4. optimizer')
    trainable_params = []
    for name, param in student_unet.named_parameters():
        prun = [block for block in skip_layers_dot if block in name]
        if len(prun) > 0:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    optimizer = torch.optim.AdamW(trainable_params,lr=1e-4)
    student_unet.train()

    print(f' step 5. make dataloader')
    csv.field_size_limit(sys.maxsize)
    csv_path = "/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv"
    video_folder = "/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample"
    datavideo_size = 512
    sample_n_frames = 16
    train_dataset = DistillWebVid10M(csv_path=csv_path,
                                     video_folder=video_folder,
                                     sample_size=datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=1,
                                                   num_workers=4,
                                                   drop_last=True)

    logger.info(f'\n step 6. LR Scheduler creation')
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.per_gpu_batch_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps, )


    print(f' step 7. accelerator prepare')
    print(f' ***** student_unet gradient checkpointing ***** ')
    teacher_unet.enable_gradient_checkpointing()
    student_unet.enable_gradient_checkpointing()
    student_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare([student_unet, optimizer, train_dataloader, lr_scheduler])

    print(f' step 8. train')
    for epoch in range(10):

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(student_unet):

                with torch.no_grad():

                    # 1. Load and process the image and text conditioning
                    pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
                    video_length = pixel_values.shape[1]
                    pixel_value = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                    pixel_value = pixel_value.to(device=device, dtype=weight_dtype)
                    latents = vae.encode(pixel_value).latent_dist.sample()  # here problem ...
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)  # [batch, channel, frame, height, width]
                    latents = latents * 0.18215  # always main process ?
                    latents = latents.to(weight_dtype)
                    bsz = latents.shape[0]

                    # 2. time step
                    topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                    index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                    start_timesteps = solver.ddim_timesteps[index] # Here Problem
                    timesteps = start_timesteps - topk
                    timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                    # 3.
                    noise = torch.randn_like(latents)
                    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                    # 4.
                    prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length,
                                           padding="max_length", truncation=True,
                                           return_tensors="pt").input_ids.to(latents.device)
                    prompt_embeds = text_encoder(prompt_ids)[0]

                    # 5.1 teacher output
                    inputs_teacher = []
                    outputs_teacher = []
                    teacher_output = teacher_unet(noisy_model_input.to(dtype=weight_dtype),
                                                  timesteps,
                                                  prompt_embeds).sample

                inputs_student = []
                outputs_student = []
                student_output = student_unet(noisy_model_input.to(dtype=weight_dtype),
                                              timesteps, prompt_embeds).sample

                # [1] distill loss
                if args.loss_type == "l2":
                    distill_loss = F.mse_loss(student_output.float(), teacher_output.float(), reduction="mean")
                elif args.loss_type == "huber":
                    distill_loss = torch.mean(torch.sqrt(
                        (student_output.float() - teacher_output.float()) ** 2 + args.huber_c ** 2) - args.huber_c)

                # [2] feature matching loss
                feature_loss = 0
                layer_num = len(inputs_teacher)
                for input_t, output_t, input_s, output_s in zip(inputs_teacher, outputs_teacher, inputs_student, outputs_student):
                    teacher_value = output_t - input_t  # Teacher 모델의 출력과 입력의 차이
                    student_value = output_s - input_s
                    feature_loss += F.mse_loss(teacher_value.float(), student_value.float()).mean()  # MSE 손실 계산
                feature_loss = feature_loss / layer_num

                inputs_teacher.clear()
                outputs_teacher.clear()
                inputs_student.clear()
                outputs_student.clear()

                # [3] total loss
                total_loss = distill_loss + args.feature_matching_loss_weight * feature_loss
                optimizer.zero_grad()
                accelerator.backward(total_loss)
                optimizer.step()

                if is_main_process:
                    print(F' [main process] total_loss = {total_loss} device = {total_loss.device}')
                    wandb.log({"distill_loss": distill_loss.detach().item(),
                               "feature_loss": feature_loss.detach().item(),
                               "total_loss": total_loss.detach().item()}, step=global_step)
                    print(f' main process, lr step')
                    lr_scheduler.step()
                    progress_bar.update(1)
                    global_step += 1

        # save model
        if is_main_process:
            save_model = accelerator.unwrap_model(student_unet)
            save_folder = os.path.join(args.output_dir, "student_model")
            os.makedirs(save_folder, exist_ok=True)
            save_dir = os.path.join(save_folder, f"student_model_{epoch+1}.pt")
            print(f' -- trained model save --')
            torch.save(save_model.state_dict(), save_dir)
            custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
            with open(custom_prompt_dir, 'r') as f:
                custom_prompts = f.readlines()




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
    parser.add_argument("--teacher_revision", type=str, default=None, required=False,
                        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.", )
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    args = parser.parse_args()
    main(args)