import sys
import os
import logging
import argparse
from accelerate import Accelerator
import torch
from accelerate import DistributedDataParallelKwargs
from diffusers.utils import check_min_version
from torch import nn
from PIL import Image as pilImage
from prun.attn.masactrl_utils import register_motion_editor
from prun.attn.controller import MotionControl
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import csv, json
from safetensors.torch import save_file
from safetensors import safe_open
import numpy as np
from prun.eval.VBench.vbench.motion_smoothness import MotionSmoothness
from prun.utils.clip_score import ClipScorer                    
from prun.data import DistillWebVid10M
from torch.utils.data import RandomSampler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from einops import rearrange
from prun.utils.diffusion_misc import * # get_predicted_original_sample
from torch.nn import functional as F
import wandb
from typing import Dict, Tuple
from diffusers import DDPMScheduler, DDIMScheduler
from prun.ode_solver import DDIMSolver

def evaluation(prompt, evaluation_pipe, controller, save_folder,
               n_prompt, num_frames, guidance_scale, num_inference_steps, h=512, weight_dtype=torch.float16):
    prompt = prompt.strip()
    with torch.no_grad() :
        output = evaluation_pipe(prompt=prompt,
                                negative_prompt=n_prompt,
                                num_frames=num_frames,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                height=h,
                                width=h,
                                generator=torch.Generator("cpu").manual_seed(args.seed),
                                dtype=weight_dtype,
                                do_use_same_latent=args.do_use_same_latent)                                
    if controller is not None :
        controller.reset()
        frames = output.frames[0]
        export_to_video(frames, os.path.join(save_folder, f'pruned_{prompt}.mp4'))
    else :
        frames = output.frames[0]
        export_to_video(frames, os.path.join(save_folder, f'teacher_{prompt}.mp4'))

def huber_loss(pred, target, huber_c=0.001):
    loss = torch.sqrt((pred.float() - target.float()) ** 2 + huber_c ** 2) - huber_c
    return loss.mean()

def kl_loss_fn(teacher_output, student_output, temperature=1.0):
    # 온도 조정을 통해 확률 분포 부드럽게 하기
    teacher_probs = F.softmax(teacher_output / temperature, dim=1)  # teacher output을 softmax
    student_log_probs = F.log_softmax(student_output / temperature, dim=1)  # student output을 log-softmax

    # KL divergence 계산
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss


def main(args):
    
    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], )
    
    is_main_process = accelerator.is_main_process
    if is_main_process:
        run = wandb.init(project='layerwise_learning', name=args.wandb_run_name, group="DDP", )
    
    weight_dtype = torch.float32
    print(f' present weight dtype = {weight_dtype}')

    logger.info(f'\n step 3. saving dir')
    guidance_scale = args.guidance_scale

    logger.info(f'\n step 5. preparing pruning')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=test_adapter, torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(test_pipe.scheduler.config,beta_schedule="linear")
    test_pipe.scheduler = noise_scheduler
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()
    
    logger.info(f'\n step 6. solver')
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(accelerator.device)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(accelerator.device)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),timesteps=noise_scheduler.config.num_train_timesteps,ddim_timesteps=args.num_ddim_timesteps,).to(accelerator.device)
    
    pruned_unet = test_pipe.unet
    #pretrained_unet_dir = r'/home/dreamyou070/Prun/pruned_unet.pth'
    pretrained_unet_dir = r'/scratch2/dreamyou070/Prun/result/timewise_permutation/sparse_model_idx/pruned_unet.pth'
    pruned_unet.load_state_dict(torch.load(pretrained_unet_dir))
    
    logger.info(f'\n step 6. preparing teacher model')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter, torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config,beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    teacher_pipe.enable_vae_slicing()
    teacher_pipe.enable_model_cpu_offload()
    #teacher_pipe.to('cuda')
    teacher_unet = teacher_pipe.unet
    vae = teacher_pipe.vae
    text_encoder = teacher_pipe.text_encoder
    tokenizer = teacher_pipe.tokenizer
    
    logger.info(f'\n step 7. pruning')    
    base_out_dir = args.output_dir
    os.makedirs(base_out_dir, exist_ok = True)
    student_out_dir = os.path.join(base_out_dir, 'pruned_model')
    os.makedirs(student_out_dir, exist_ok = True)
    teacher_out_dir = os.path.join(base_out_dir, 'teacher_model')
    os.makedirs(teacher_out_dir, exist_ok = True)
        
    pruned_unet = test_pipe.unet
    motion_controller = MotionControl(guidance_scale=guidance_scale,
                                      window_attention=False,
                                      skip_layers=args.skip_layers,
                                      is_teacher=False,
                                      batch_size=1,
                                      train=False,
                                      pruned_unorder=args.pruned_unorder,
                                      self_attn_pruned = args.self_attn_pruned)
    # check unet state_dictionary
    original_params_dict = dict(teacher_unet.named_parameters())
    pruned_params_dict = dict(pruned_unet.named_parameters())
    # basic settnig for pruned_unet -> requires_grad = False
    pruned_unet.requires_grad = False
    params_to_train = []
    
    for name, parameter in pruned_unet.named_parameters():    
        original_param = original_params_dict[name]
        pruned_param = pruned_params_dict[name]
        if 'motion' in name and not torch.equal(original_param, pruned_param) :
            
            if (pruned_param == 0).all():
                pruned_param.requires_grad = False            
            
            else :
                
                if original_param.dim() == 2 :
                    out_dim, in_dim = original_param.shape
                    for i in range(out_dim):
                        org_row = original_param[i,:]
                        spr_row = pruned_param[i,:]
                        if (pruned_param == 0).all():
                            pruned_param[i,:].requires_grad = False
                        else :
                            if torch.equal(org_row, spr_row):
                                pruned_param[i,:].requires_grad = False
                            else :
                                print(f'param to train 1')
                                pruned_param[i,:] = pruned_param[i,:] * 0                                
                                pruned_param[i,:].requires_grad = True
                                params_to_train.append(pruned_param[i,:])
                else :
                    out_dim = original_param.shape[0]
                    for i in range(out_dim):
                        org_row = original_param[i]
                        spr_row = pruned_param[i]
                        if torch.equal(org_row, spr_row) :
                            pruned_param[i].requires_grad = False
                        else :
                            print(f'param to train 2')
                            pruned_param[i] = pruned_param[i] * 0
                            pruned_param[i].requires_grad = True
                            params_to_train.append(pruned_param[i])
    params_to_train = [p for p in pruned_unet.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_train, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    
    # make dataset and dataloader
    logger.info(f' step 8. data loader')
    csv.field_size_limit(sys.maxsize)
    args.csv_path = r'/home/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv'
    args.video_folder = r'/home/dreamyou070/MyData/video/panda/test_sample_trimmed/sample'
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size, sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, drop_last=True)
        
    logger.info(f'\n step 9. set optimizer')
    logger.info(f' (9.1) mask object training')
    logger.info(f' (9.2) pruned unet target layer linear module')
    logger.info(f'\n step 10. lr scheduler')
    
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)
    
    logger.info(f'\n step 11. prepare')
    
    # [1] accelerator
    teacher_unet, pruned_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(teacher_unet, pruned_unet, optimizer, train_dataloader, lr_scheduler)
    vae, text_encoder = vae.to(accelerator.device, dtype=weight_dtype), text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # [2] scaler
    logger.info(f'\n step 12. Train')
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}") # 10
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}") # 1
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Len train_dataloader = {len(train_dataloader)}")
    
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",disable=not accelerator.is_local_main_process,)
    global_step = 0
    
    for epoch in range(args.first_epoch, args.num_train_epochs):

        pruned_unet.train()
        teacher_unet.eval()
        
        for step, batch in enumerate(train_dataloader):
            
            # [1] : dataloader multi gpu
            optimizer.zero_grad()
            
            pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
            video_length = pixel_values.shape[1]
            pixel_value = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(accelerator.device, dtype=weight_dtype)
            
            latents = vae.encode(pixel_value).latent_dist.sample() # here problem ...
            latents = rearrange(latents, "(b f) c h w -> b c f h w",f=video_length)  # [batch, channel, frame, height, width]
            latents = latents * 0.18215 # always main process ?
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # [2] time embedding
            topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,)).long()
            start_timesteps = solver.ddim_timesteps[index] # on device ...
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

            # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(timesteps)
            c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
            noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
            
            # [2] text encoding
            prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(latents.device)
            encoder_hidden_states = text_encoder(prompt_ids)[0]
            # encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
            
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            with torch.no_grad():
                teacher_unet = teacher_unet.to(noisy_model_input.device)
                teacher_output = teacher_unet(noisy_model_input, # sample
                                              timesteps,
                                              encoder_hidden_states).sample
                teacher_pred_x_0 = get_predicted_original_sample(teacher_output,
                                                                 start_timesteps,
                                                                 noisy_model_input,
                                                                 "epsilon",
                                                                 alpha_schedule,
                                                                 sigma_schedule, )
                teacher_model_pred = c_skip_start * noisy_model_input + c_out_start * teacher_pred_x_0  # consistency                
            student_output = pruned_unet(noisy_model_input, timesteps, encoder_hidden_states).sample  # [1,4,16,h,w]
            
            # [1] vlb loss  
            vlb_loss = F.mse_loss(student_output.float(), target.float(), reduction = 'mean')
            
            # [2] 
            motion_controller.reset()            
            student_pred_x_0 = get_predicted_original_sample(student_output, start_timesteps, noisy_model_input,
                                                             "epsilon", alpha_schedule,  sigma_schedule, )  # [1,4,16,h,w]
            
            student_model_pred = c_skip_start.to(student_pred_x_0.device) * noisy_model_input.to(student_pred_x_0.device) 
            + c_out_start.to(student_pred_x_0.device) * student_pred_x_0  # consistency
            
            # [1] distil loss (not use)
            if args.loss_type == "l2":
                distill_loss = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
            elif args.loss_type == "huber":
                distill_loss = huber_loss(student_model_pred, teacher_model_pred, args.huber_c)
            
            # [3] mu loss (why it change ??)
            mu_t = teacher_model_pred.mean(dim=(1, 2))
            mu_s = student_model_pred.mean(dim=(1, 2))
            mu_loss = F.mse_loss(mu_t.to(mu_s.device).float(),mu_s.float(), reduction="mean")
            
            # [4] fvd loss
            loss =  mu_loss * args.mu_loss_scale
            # [5] KL Divergence Loss
            kl_loss = kl_loss_fn(teacher_output, student_output) 
            if args.use_kl_loss :
                loss = loss + kl_loss.mean()
            
            accelerator.backward(loss)
            
            # [6] optimizer step
            torch.nn.utils.clip_grad_norm_(params_to_train, args.max_grad_norm)
            #optimizer.step()
            #lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            accelerator.wait_for_everyone()
            
            if is_main_process:
                # loss = vlb_loss + use_ratio_loss * args.use_ratio_loss_scale + mu_loss * args.mu_loss_scale
                #wandb.log({" vlb_loss": vlb_loss.item()})
                wandb.log({" mu_loss": mu_loss.item()})
                if args.use_kl_loss :
                    wandb.log({" kl_loss": kl_loss.item()})
                #if args.reward_fn:
                #    wandb.log({" aesthetic_loss": aesthetic_loss.item()})
                #    wandb.log({" clip_loss": clip_loss.item()})
        accelerator.wait_for_everyone()
        
        # save model
        if is_main_process:
            # Periodically validation
            save_pruning_object2 = accelerator.unwrap_model(pruned_unet)
            save_path = os.path.join(base_out_dir, f"pruned_unet_checkpoints")
            os.makedirs(save_path, exist_ok=True)
            state_dict = {"epoch": epoch, "global_step": global_step, "state_dict": save_pruning_object2.state_dict(), }
            torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch + 1}.ckpt"))
            logging.info(f"Saved state to {save_path} (global_step: {global_step})")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='video_distill')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument('--sub_folder_name', type=str, default='result_sy')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", default='fp16')
    parser.add_argument('--full_attention', action='store_true')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--motion_control', action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    from prun.utils import arg_as_list

    parser.add_argument('--skip_layers', type=arg_as_list, default=[])
    parser.add_argument('--skip_layers_dot', type=arg_as_list, default=[])
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--inference_step', type=int, default=6)
    parser.add_argument('--csv_path', type=str, default='data/webvid-10M.csv')
    parser.add_argument('--video_folder', type=str, default='data/webvid-10M')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--launcher', type=str)
    parser.add_argument('--cfg_random_null_text', action='store_true')
    parser.add_argument('--cfg_random_null_text_ratio', type=float, default=1e-1)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=100)
    parser.add_argument('--saved_epoch', type=int, default=16)
    parser.add_argument('--prompt_file_dir', type=str, default=r'configs/validation/prompt700.txt')
    parser.add_argument("--do_teacher_inference", action='store_true')
    parser.add_argument("--do_raw_pruning_teacher", action='store_true')
    parser.add_argument("--do_training_test", action='store_true')
    parser.add_argument("--attention_reshaping_test", action='store_true')
    parser.add_argument("--model_base_dir", type=str)
    parser.add_argument("--training_test", action='store_true')
    parser.add_argument('--patch_len', type=int, default=3)
    parser.add_argument('--window_attention', action='store_true')
    parser.add_argument('--num_inference_step', type=int)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--vbench_dir", type=str)
    parser.add_argument("--training_modules", type=str)
    parser.add_argument("--frame_num", type=int, default=16)
    parser.add_argument("--lora_training", action='store_true')
    parser.add_argument("--motion_base_folder", type=str)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--do_save_attention_map', action='store_true')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoints')
    parser.add_argument('--do_first_frame_except', action='store_true')
    parser.add_argument('--teacher_student_interpolate', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--training_test_merging', action='store_true')
    parser.add_argument("--target_change_index", type=int, default=128)
    parser.add_argument('--add_qkv', action='store_true')
    parser.add_argument('--inference_only_front_motion_module', action='store_true')
    parser.add_argument('--inference_only_post_motion_module', action='store_true')
    parser.add_argument('--pruning_test', action='store_true')

    parser.add_argument('--do_use_same_latent', action='store_true')
    parser.add_argument('--do_teacher_train_mode', action='store_true')
    parser.add_argument('--pruned_unorder', action='store_true')
    parser.add_argument('--self_attn_pruned', action='store_true')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--pruning_ratio_list', type=arg_as_list, default=[])
    parser.add_argument("--wandb_run_name", type=str)
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
    parser.add_argument("--ddim_eta", type=float, default=0.0, help=("Eta for solving the DDIM step."), )
    parser.add_argument("--datavideo_size", type=int, default=128)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument('--noise_scheduler_kwargs', type=Dict)
    parser.add_argument('--max_train_epoch', type=int, default=-1)
    parser.add_argument('--max_train_steps', type=int, default=-1)
    parser.add_argument('--validation_steps', type=int, default=100)
    parser.add_argument('--validation_steps_tuple', type=Tuple, default=(-1,))
    parser.add_argument('--scale_lr', action='store_true')
    
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--trainable_modules', type=arg_as_list, default="['motion_modules.']")
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--checkpointing_epochs', type=int, default=5)
    parser.add_argument('--checkpointing_steps', type=int, default=-1)
    parser.add_argument('--mixed_precision_training', action='store_true')
    parser.add_argument('--is_debug', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.", )
    parser.add_argument("--do_window_attention", action="store_true")
    parser.add_argument("--beta_schedule",
                        default="scaled_linear",
                        type=str,
                        help="The schedule to use for the beta values.", )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--first_epoch', type=int, default=0)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="huber",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument("--mu_loss_scale",
        type=float,
        default=1.0,
        help="The scale of the reward loss",
    )
    parser.add_argument('--mask_do_train', action="store_true")
    parser.add_argument('--use_kl_loss', action="store_true")   
    args = parser.parse_args()
    main(args)
