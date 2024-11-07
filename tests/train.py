""" compare between retraining and distillation """
# Temporal Attention Mechanisms:
# Attention 메커니즘을 사용하여 중요한 시간적 패턴을 distillation할 수 있습니다. 비디오의 각 프레임이 가진 motion 정보를 강조하는 attention 메커니즘을 활용하는 것입니다.
# 이를 통해 학생 모델이 중요한 motion 정보를 학습하도록 할 수 있습니다.
# Implementation: Teacher 모델의 attention map을 학생 모델이 재현하도록 학습시키는 것이 한 방법입니다.

# Optical Flow 기반 Distillation:
# Optical Flow는 비디오에서의 motion 정보를 추출하는 대표적인 방법입니다.
# Teacher 모델의 프레임들 사이에서 Optical Flow를 계산하고, 학생 모델이 이 Optical Flow 패턴을 모방하도록 학습시키는 방법입니다.
# Implementation:
import sys
import os
import math
import wandb
import random
import logging
import inspect
import gc
import argparse
from torch.utils.data import RandomSampler
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from typing import Dict
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from prun.models.motion import MotionAdapter
from prun.models.pipelines import AnimateDiffPipeline
from prun.models.scheduler import LCMScheduler
from prun.data import DistillWebVid10M
from prun.utils.layer_dictionary import find_layer_name, find_next_layer_name
from prun.attn.masactrl_utils import register_motion_editor
from prun.attn.controller import MotionControl
from prun.utils.accelerator_utils import make_accelerator, get_folder, make_skip_layers_dot
from diffusers.utils import export_to_gif, export_to_video, load_image
import GPUtil
import json
from diffusers import DDPMScheduler, DDIMScheduler
from prun.utils.diffusion_misc import *
from diffusers.video_processor import VideoProcessor
import numpy as np
import torchvision
import imageio
import yaml
from safetensors.torch import save_file
from safetensors import safe_open
from prun.utils import arg_as_list
from prun.ode_solver import DDIMSolver
from torch import nn


def evaluateion(prompt, evaluation_pipe, controller, save_folder,
                n_prompt, num_frames, guidance_scale, num_inference_steps):
    prompt = prompt.strip()
    output = evaluation_pipe(prompt=prompt,
                             negative_prompt=n_prompt,
                             num_frames=num_frames,
                             guidance_scale=guidance_scale,
                             num_inference_steps=num_inference_steps,
                             generator=torch.Generator("cpu").manual_seed(args.seed), )
    controller.reset()
    frames = output.frames[0]
    export_to_gif(frames, os.path.join(save_folder, f'{prompt}-0.gif'))
    export_to_video(frames, os.path.join(save_folder, f'{prompt}-0.mp4'))
    text_dir = os.path.join(save_folder, f'{prompt}-0.txt')
    with open(text_dir, 'w') as f:
        f.write(f'prompt : {prompt}\n')
        f.write(f'n_prompt : {n_prompt}\n')
        f.write(f'guidance_scale : {guidance_scale}\n')
        f.write(f'num_inference_steps : {num_inference_steps}\n')
        f.write(f'seed : {args.seed}\n')


def decode_latents(vae, latents):
    latents = 1 / vae.config.scaling_factor * latents

    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

    image = vae.decode(latents).sample
    video = image[None, :].reshape((batch_size, num_frames, -1) + image.shape[2:]).permute(0, 2, 1, 3, 4)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    video = video.float()
    return video


def load_img(img):
    rgb_img = np.array(img, np.float32).squeeze()
    img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()

    img_tensor = (img_tensor / 255. - 0.5) * 2
    return img_tensor


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)
    return kohya_ss_state_dict


def main(args):

    torch.distributed.init_process_group(backend='nccl')

    GPUtil.showUtilization()
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 0. basic setting')
    guidance_scale = args.guidance_scale
    num_frames = args.num_frames
    num_inference_steps = args.inference_step
    n_prompt = "bad quality, worse quality, low resolution"
    custom_prompt_dir = r'/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        custom_prompts = f.readlines()


    logger.info(f'\n step 1. setting')
    torch.manual_seed(args.seed)
    weight_dtype = torch.float32

    logger.info(f'\n step 2. accelerator and file')
    accelerator = make_accelerator(args)
    is_main_process = accelerator.is_main_process
    output_dir, custom_save_folder, eval_save_folder, sanity_folder, model_save_dir, log_folder = get_folder(
        args.output_dir,
        args.sub_folder_name)

    logger.info(f'\n step 4. get teacher model')
    device = accelerator.device
    noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                    subfolder="scheduler",
                                                    beta_schedule=args.beta_schedule, )
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)
    scale_b = 0.7
    use_scale = False
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        ddim_timesteps=args.num_ddim_timesteps,
                        use_scale=use_scale,
                        scale_b=scale_b,
                        ddim_eta=args.ddim_eta, )
    solver = solver.to(device)

    print(f' step 6. pretrained_teacher_model')
    teacher_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
    if args.pretrained_teacher_adapter:
        pretrained_state_dict = torch.load(args.pretrained_teacher_adapter, map_location="cpu")

    args.pretrained_model_path = "emilianJR/epiCRealism"
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=teacher_adapter,
                                                       torch_dtpe=weight_dtype)
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                   weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    teacher_unet = teacher_pipe.unet
    teacher_unet.requires_grad_(False)
    teacher_unet.to(device, dtype=weight_dtype)
    teacher_pipe.scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")

    ################################################     

    teacher_motion_controller = MotionControl(guidance_scale=guidance_scale,
                                              # frame_num=16,
                                              window_attention=False,
                                              skip_layers=[],
                                              is_teacher=True,
                                              train=True,
                                              do_save_attention_map=args.do_save_attention_map)  # 32
    register_motion_editor(teacher_unet, teacher_motion_controller)

    logger.info(f'\n step 5. other models')
    tokenizer = teacher_pipe.tokenizer
    vae = teacher_pipe.vae
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder = teacher_pipe.text_encoder
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    logger.info(f'\n step 6. student model')
    student_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype).to(device)
    student_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                       motion_adapter=student_adapter,
                                                       torch_dtpe=weight_dtype)
    student_unet = student_pipe.unet
    student_unet.requires_grad_(False)
    student_unet.to(device, dtype=weight_dtype)  # this cannot be ?
    guidance_scale = args.guidance_scale
    student_motion_controller = MotionControl(guidance_scale=guidance_scale,
                                              # frame_num=16,
                                              window_attention=args.window_attention,
                                              skip_layers=args.skip_layers,
                                              is_teacher=False,
                                              train=True,
                                              do_save_attention_map=args.do_save_attention_map)
    register_motion_editor(student_unet, student_motion_controller)
    if args.pretrained_student_adapter:
        pretrained_state_dict = torch.load(args.pretrained_student_adapter, map_location="cpu")
        for k, v in student_unet.named_parameters():
            if 'motion' in k:
                student_unet.state_dict()[k].copy_(pretrained_state_dict[k])
    student_pipe.motion_module = student_adapter
    args.lora_rank = 64
    # lora_config = LoraConfig(r=args.lora_rank,
    #                         target_modules=["to_q","to_k","to_v","to_out.0","proj_in","proj_out", "ff.net.0.proj","ff.net.2","conv1","conv2",
    #                                         "conv_shortcut","downsamplers.0.conv","upsamplers.0.conv","time_emb_proj",],)
    # student_unet = get_peft_model(student_unet, lora_config)

    student_pipe.unet = student_unet
    student_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                   weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")  # train with LCM lora
    student_pipe.set_adapters(["lcm-lora"], [0.8])
    student_pipe.scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")

    logger.info(f'\n step 7. scale lr')
    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.per_gpu_batch_size)

    logger.info(f'\n step 8. optimzier')
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    parameters_list = []

    args.skip_layers = ['up_blocks_0_motion_modules_0']
    skip_layers_dot = make_skip_layers_dot(args.skip_layers)
    student_unet.requires_grad_(False)
    for name, para in student_unet.named_parameters():
        if 'motion' in name:
            para.requires_grad = True  # training only some layers
            for skip_layer in skip_layers_dot:  # not graedint
                if skip_layer in name:
                    para.requires_grad = False
                    break
        elif 'lora' in name:
            if args.lora_training:
                para.requires_grad = True
            else:
                para.requires_grad = False
        else:
            para.requires_grad = False

    for name, para in student_unet.named_parameters():
        if para.requires_grad:
            parameters_list.append(para)
    param_groups = []
    param_groups.append({'params': parameters_list, 'lr': args.learning_rate})
    optimizer = optimizer_cls(param_groups,
                              betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay,
                              eps=args.adam_epsilon, )

    logger.info(f'\n step 9. gradient check pointing')
    if args.gradient_checkpointing:
        student_unet.enable_gradient_checkpointing()

    logger.info(f'\n step 10. data loader')
    import csv
    csv.field_size_limit(sys.maxsize)
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=True)

    logger.info(f'\n step 11. training steps')
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    print(f' step 12. learning rate scheduler')
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)

    print(f' step 13. sanitary check')
    for step, batch in enumerate(train_dataloader):
        pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
        for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
            pixel_value = pixel_value[None, ...]
            save_videos_grid(pixel_value, os.path.join(sanity_folder, f'data_sample_check.gif'), rescale=True)
        break

    print(f' step 15. Before Training Teacher Inference')
    if args.do_teacher_inference:
        f_1 = os.path.join(custom_save_folder, 'teacher')
        os.makedirs(f_1, exist_ok=True)
        for p, prompt in enumerate(custom_prompts):
            evaluateion(prompt, teacher_pipe, teacher_motion_controller, f_1, n_prompt, num_frames, guidance_scale,
                        num_inference_steps)

        f_2 = os.path.join(eval_save_folder, 'teacher')
        os.makedirs(f_2, exist_ok=True)


    print(f' step 15. Train!')
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print(f'args.max_train_steps : {args.max_train_steps}')
    print(f'num_update_steps_per_epoch : {num_update_steps_per_epoch}')
    print(f' args.num_train_epochs : {args.num_train_epochs}')

    total_batch_size = args.per_gpu_batch_size * args.gradient_accumulation_steps  # 300
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")
    vae_scale_factor = 0.18215
    first_epoch = args.first_epoch

    logger.info(f'\n step 16. save argument')
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # preparing
    student_unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(student_unet, optimizer, lr_scheduler,
                                                                                  train_dataloader)

    first_epoch, args.num_train_epochs = 0, 10
    for epoch in range(first_epoch, args.num_train_epochs):
        teacher_unet.eval()
        student_unet.train()
        for step, batch in enumerate(train_dataloader):

            infer = False

            if epoch == first_epoch:
                if args.do_raw:
                    infer = True
            else:
                infer = True

            if step == 0 and infer:
                accelerator.wait_for_everyone()
                # if is_main_process:

                with torch.no_grad():

                    # [1] get trained motion adapter
                    eval_adapter = MotionAdapter.from_pretrained(args.teacher_motion_model_dir, torch_dtpe=weight_dtype)
                    eval_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path,
                                                                    motion_adapter=eval_adapter,
                                                                    torch_dtpe=weight_dtype)
                    eval_unet = eval_pipe.unet
                    eval_motion_controller = MotionControl(guidance_scale=guidance_scale,  # frame_num=16,
                                                           window_attention=args.window_attention,
                                                           skip_layers=args.skip_layers, train=False,
                                                           is_teacher=False, )
                    register_motion_editor(eval_unet, eval_motion_controller, args)
                    # [3.1] make pipe (+ motion module)                        
                    eval_pipe.load_lora_weights("wangfuyun/AnimateLCM",
                                                weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                                adapter_name="lcm-lora")
                    eval_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM

                    # **************** TODO **************** #
                    eval_trained_value = accelerator.unwrap_model(student_unet).state_dict()
                    # eval_trained_value = student_unet.state_dict()
                    eval_state_dict = {}
                    for trained_key, trained_value in eval_trained_value.items():
                        if 'motion' in trained_key:
                            eval_unet.state_dict()[trained_key].copy_(trained_value.to('cpu'))

                    # [2] get trained lora
                    if args.lora_training:
                        lora_state_dict = {}
                        for trained_key, trained_value in eval_trained_value.items():
                            if 'lora' in trained_key:
                                eval_unet.state_dict()[trained_key].copy_(trained_value.to('cpu'))

                    eval_pipe.unet = eval_unet
                    eval_unet.to(device, dtype=weight_dtype)
                    # [3.6] scheduler
                    eval_pipe.scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config,
                                                                   beta_schedule="linear")
                    eval_pipe.to(device)
                    for p, prompt in enumerate(custom_prompts):
                        f_1 = os.path.join(custom_save_folder, f'student_epoch_{str(epoch).zfill(3)}')
                        os.makedirs(f_1, exist_ok=True)
                        evaluateion(prompt, eval_pipe, eval_motion_controller, f_1, n_prompt, num_frames,
                                    guidance_scale, num_inference_steps)

                    del eval_pipe, eval_unet, eval_motion_controller, eval_adapter, eval_trained_value, eval_state_dict
            # accelerator.wait_for_everyone()

            if step == 0:
                accelerator.wait_for_everyone()
                if is_main_process:
                    # **************** TODO **************** #
                    student_state_dict = accelerator.unwrap_model(student_unet).state_dict()
                    # student_state_dict = student_unet.state_dict()
                    # [1] 
                    save_state_dict = {}
                    for trained_key, trained_value in student_state_dict.items():
                        if 'motion' in trained_key:
                            save_state_dict[trained_key] = trained_value.to('cpu')
                    # [2] 
                    if args.lora_training:
                        lora_state_dict = {}
                        for trained_key, trained_value in student_state_dict.items():
                            if 'lora' in trained_key:
                                lora_state_dict[trained_key] = trained_value.to('cpu')

                    save_epoch = str(epoch).zfill(3)
                    motion_module_folder = os.path.join(output_dir, f"checkpoints/motion_module")
                    os.makedirs(motion_module_folder, exist_ok=True)
                    torch.save(save_state_dict,
                               os.path.join(output_dir, f"checkpoints/motion_module/checkpoint_epoch_{save_epoch}.pt"))
                    if args.lora_training:
                        lora_folder = os.path.join(output_dir, f"checkpoints/lcm_lora")
                        os.makedirs(lora_folder, exist_ok=True)
                        torch.save(lora_state_dict,
                                   os.path.join(output_dir, f"checkpoints/lcm_lora/checkpoint_epoch_{save_epoch}.pt"))
                accelerator.wait_for_everyone()
            # ------------------------------------------------------------------------------------------------------------
            # register motion controller            
            # [1]
            pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
            video_length = pixel_values.shape[1]
            b, t = pixel_values.shape[0], pixel_values.shape[1]
            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                with torch.no_grad():
                    latents = vae.encode(pixel_values.to(device, dtype=weight_dtype)).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w",
                                        f=video_length)  # [batch, channel, frame, height, width]
                latents = latents * 0.18215

            # [2] text encoding
            with torch.no_grad():
                # prompt = batch['text']
                prompt_ids = tokenizer(
                        batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                        return_tensors="pt").input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]

            bsz = latents.shape[0]
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - args.topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
            # 3.1 Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps,
                                                                         timestep_scaling=args.timestep_scaling_factor)
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=args.timestep_scaling_factor)
            c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
            noise = torch.randn_like(latents)
            noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            ##################################################################################################
            # [5]
            with torch.no_grad():
                teacher_output = teacher_unet(noisy_model_input, timesteps, encoder_hidden_states).sample
                if args.motion_control and args.do_save_attention_map:
                    t_attention_map_dict = teacher_motion_controller.attnmap_dict
                    teacher_motion_controller.reset()
                    teacher_motion_controller.attnmap_dict = {}

            student_output = student_unet(noisy_model_input, timesteps, encoder_hidden_states).sample

            if args.motion_control and args.do_save_attention_map:
                s_attention_map_dict = student_motion_controller.attnmap_dict
                student_motion_controller.reset()
                student_motion_controller.attnmap_dict = {}

            # consistent student model
            pred_x_0 = get_predicted_original_sample(student_output,
                                                     start_timesteps,
                                                     noisy_model_input,
                                                     "epsilon",
                                                     alpha_schedule,
                                                     sigma_schedule, )
            student_model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0  # consistency

            ##################################################################################################
            # [6] lossess
            distill_loss = torch.zeros_like(student_model_pred).mean()
            feature_matching_loss = 0
            ##################################################################################################
            if args.do_save_attention_map:
                for layer_name in t_attention_map_dict.keys():
                    t_h = t_attention_map_dict[layer_name]
                    s_h = s_attention_map_dict[layer_name]
                    for t_h_, s_h_ in zip(t_h, s_h):
                        feature_matching_loss += F.mse_loss(s_h_.float(), t_h_.float(), reduction="mean")

            # [6.3] distillation

            if args.framewise_distill:
                frame_num = student_output.shape[2]
                for i in range(frame_num):
                    distill_loss += F.mse_loss(student_output[:, :, i].float(), teacher_output[:, :, i].float(),
                                               reduction="mean")
            elif args.edge_distill:
                distill_loss += F.mse_loss(student_output[:, :, -1].float(), teacher_output[:, :, -1].float(),
                                           reduction="mean")
            else:
                distill_loss = F.mse_loss(student_output.float(), teacher_output.float(), reduction="mean")

            if args.gradient_learning:
                teacher_gradients = []
                student_gradients = []
                # [1] teacher frame diff (1-0, 2-1, ..., )
                frame_diffs_teacher = teacher_output[:, :, 1:, :, :] - teacher_output[:, :, :-1, :, :]
                # [2] student frame diff (1-0, 2-1, ..., )
                frame_diffs_student = student_output[:, :, 1:, :, :] - student_output[:, :, :-1, :, :]
                # [3] gradient should be same direction
                gradient_loss = F.mse_loss(frame_diffs_student, frame_diffs_teacher.detach())

            if args.do_distill:
                total_loss = args.feature_matching_weight * feature_matching_loss + args.distill_weight * distill_loss
                if args.gradient_learning:
                    total_loss += args.gradient_weight * gradient_loss.mean()
            else:
                if args.gradient_learning:
                    total_loss = args.gradient_weight * gradient_loss.mean()
                else:
                    assert False, 'no loss'

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            if args.motion_control:
                student_motion_controller.reset()
                teacher_motion_controller.reset()
            accelerator.wait_for_everyone()

        if args.motion_control:
            student_motion_controller.reset()
            teacher_motion_controller.reset()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default='fp16')
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--skip_layers', type=arg_as_list)
    parser.add_argument('--skip_layers_dot', type=arg_as_list)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--inference_step', type=int, default=6)
    parser.add_argument('--csv_path', type=str, default='data/webvid-10M.csv')
    parser.add_argument('--video_folder', type=str, default='data/webvid-10M')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--name', type=str, default='video_distill')
    parser.add_argument('--output_dir', type=str, default='experiment')
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--cfg_random_null_text', action='store_true')
    parser.add_argument('--cfg_random_null_text_ratio', type=float, default=0.1)
    parser.add_argument('--unet_checkpoint_path', type=str, default='')
    parser.add_argument('--unet_additional_kwargs', type=Dict)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--noise_scheduler_kwargs', type=Dict)
    parser.add_argument('--max_train_epoch', type=int, default=-1)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--validation_steps_tuple', type=Tuple, default=(-1,))
    parser.add_argument('--scale_lr', action='store_true')
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--trainable_modules', type=arg_as_list, default="['motion_modules.']")
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--checkpointing_epochs', type=int, default=5)
    parser.add_argument('--checkpointing_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mixed_precision_training', action='store_true')
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true')
    parser.add_argument('--is_debug', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.", )
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--do_window_attention", action="store_true")
    parser.add_argument("--datavideo_size", type=int, default=512)
    parser.add_argument(
            "--beta_schedule",
            default="scaled_linear",
            type=str,
            help="The schedule to use for the beta values.",
            )
    parser.add_argument("--do_aesthetic_loss", action='store_true', )
    parser.add_argument("--aesthetic_score_weight", type=float, default=0.5)
    parser.add_argument("--do_t2i_loss", action='store_true')
    parser.add_argument("--do_hps_loss", action='store_true')
    parser.add_argument("--clip_flant5_score", action='store_true')

    parser.add_argument("--hps_version", type=str, default="v2.1", help="hps version: 'v2.0', 'v2.1'")

    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--up_module_attention", action='store_true')
    parser.add_argument("--down_module_attention", action='store_true')
    parser.add_argument(
            "--num_ddim_timesteps",
            type=int,
            default=50,
            help="Num timesteps for DDIM sampling",
            )
    parser.add_argument(
            "--topk",
            type=int,
            default=20,
            help="1000 (Num Train timesteps) // 50 (Num timesteps for DDIM sampling)",
            )
    parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help=("Eta for solving the DDIM step."),
            )
    parser.add_argument("--feature_matching_weight", type=float, default=1.0)
    parser.add_argument("--reward_fn_name", type=str, default="hpsv2", help="Reward function name", )

    parser.add_argument("--image_reward_weight", type=float, default=1.0)
    parser.add_argument("--do_video_reward", action="store_true", help="Whether to use image reward", )
    parser.add_argument("--video_rm_batch_size", type=int, default=8,
                        help="Num frames for inputing to the text-video RM.", )
    parser.add_argument(
            "--timestep_scaling_factor",
            type=float,
            default=10.0,
            help=(
                "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
                " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
                " suffice."
            ),
            )
    parser.add_argument("--do_teacher_inference", action="store_true")
    parser.add_argument("--do_raw", action="store_true")
    parser.add_argument("--video_reward_weight", type=float, default=1.0)
    parser.add_argument("--video_rm_name", type=str)
    parser.add_argument("--video_rm_ckpt_dir", type=str)
    parser.add_argument("--do_save_attention_map", action="store_true")
    parser.add_argument("--framewise_distill", action="store_true")
    parser.add_argument("--pretrained_student_adapter", type=str)
    parser.add_argument("--pretrained_teacher_adapter", type=str)
    parser.add_argument("--first_epoch", type=int)
    parser.add_argument("--lora_training", action="store_true", help="Whether to use image reward", )
    parser.add_argument("--edge_distill", action="store_true")
    parser.add_argument("--gradient_weight", type=float, default=1.0)
    parser.add_argument("--gradient_learning", action="store_true")
    parser.add_argument("--do_distill", action="store_true")
    args = parser.parse_args()
    # name = Path(args.config).stem
    main(args)