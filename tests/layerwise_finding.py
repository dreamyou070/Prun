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


def main(args):

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], )
    weight_dtype = torch.float32
    print(f' present weight dtype = {weight_dtype}')

    logger.info(f'\n step 4. saving dir')
    n_prompt = "bad quality, worse quality, low resolution"
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_step
    custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        custom_prompts = f.readlines()

    logger.info(f'\n step 5. preparing pruning')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=test_adapter,
                                                    torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(test_pipe.scheduler.config, beta_schedule="linear")
    test_pipe.scheduler = noise_scheduler
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()

    logger.info(f'\n step 6. preparing teacher model')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter,
                                                       torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                   adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    teacher_pipe.enable_vae_slicing()
    teacher_pipe.enable_model_cpu_offload()
    teacher_pipe.to('cuda')
    teacher_unet = teacher_pipe.unet
    vae = teacher_pipe.vae
    text_encoder = teacher_pipe.text_encoder
    tokenizer = teacher_pipe.tokenizer

    logger.info(f'\n step 6. solver')
    noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(accelerator.device)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(accelerator.device)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps, ).to(accelerator.device)

    logger.info(f'\n step 7. pruning')
    base_out_dir = args.output_dir
    os.makedirs(base_out_dir, exist_ok=True)
    student_out_dir = os.path.join(base_out_dir, 'pruned_model')
    os.makedirs(student_out_dir, exist_ok=True)
    teacher_out_dir = os.path.join(base_out_dir, 'teacher_model')
    os.makedirs(teacher_out_dir, exist_ok=True)

    logger.info(f'\n step 8. dataset')
    csv.field_size_limit(sys.maxsize)
    args.csv_path = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv'
    args.video_folder = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample_'
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, drop_last=True)

    logger.info(f'\n step 8. scoring models')
    config = '/home/dreamyou070/Prun/src/prun/eval/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'
    ckpt = '/home/dreamyou070/.cache/vbench/amt_model/amt-s.pth'
    smoothness_model = MotionSmoothness(config=config, ckpt=ckpt, device='cuda')
    # [2] clip model
    clip_scorer = ClipScorer(device="cuda")
    # [3] aesthetic model


    logger.info(f'\n step 8. layer check')
    pruned_unet = test_pipe.unet
    pruned_list = []
    for layer_name, layer in pruned_unet.named_modules():
        if 'motion' in layer_name :
            if layer.__class__.__name__ == 'Linear':
                pruned_list.append(layer_name)
    state_dict = pruned_unet.state_dict()

    # most bad layer = layer_down_blocks.0.motion_modules.0.transformer_blocks.0.attn2.to_k

    for pruned_layer in pruned_list:
        # [1] zero weight and bias of pruned layer
        state_dict[pruned_layer + '.weight'] = torch.zeros_like(state_dict[pruned_layer + '.weight'])
        try :
            state_dict[pruned_layer + '.bias'] = torch.zeros_like(state_dict[pruned_layer + '.bias'])
        except:
            pass
        pruned_unet.load_state_dict(state_dict)

        # [2] make_folder
        pruned_layer_folder = os.path.join(student_out_dir, f'layer_{pruned_layer}')
        os.makedirs(pruned_layer_folder, exist_ok=True)

        # [3] check the loss
        for step, batch in enumerate(train_dataloader):

            if step < 100 :

                # [1] : dataloader multi gpu
                #optimizer.zero_grad()

                pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
                video_length = pixel_values.shape[1]
                pixel_value = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(accelerator.device, dtype=weight_dtype)

                latents = vae.encode(pixel_value).latent_dist.sample()  # here problem ...
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)  # [batch, channel, frame, height, width]
                latents = latents * 0.18215  # always main process ?

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # [2] time embedding
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,)).long()
                start_timesteps = solver.ddim_timesteps[index]  # on device ...
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # [2] text encoding
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length",
                                       truncation=True, return_tensors="pt").input_ids.to(latents.device)
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
                    teacher_output = teacher_unet(noisy_model_input,timesteps,encoder_hidden_states).sample
                    teacher_pred_x_0 = get_predicted_original_sample(teacher_output,
                                                                     start_timesteps,
                                                                     noisy_model_input,
                                                                     "epsilon",
                                                                     alpha_schedule,
                                                                     sigma_schedule)
                    teacher_latents = teacher_output * c_out + c_skip * noisy_model_input

                # make video
                student_output = pruned_unet(noisy_model_input, timesteps, encoder_hidden_states).sample
                student_pred_x_0 = get_predicted_original_sample(student_output,
                                                                 start_timesteps,
                                                                 noisy_model_input,
                                                                 "epsilon",
                                                                 alpha_schedule,
                                                                 sigma_schedule)
                denoised_latents = student_output * c_out + c_skip * noisy_model_input

                # [1] vlb loss
                # if it is normal noise from motion_controller.reset()
                if args.vlb_loss :
                    vlb_loss = F.mse_loss(student_output.float(), noise.float(), reduction='mean')
                    loss = vlb_loss
                else:
                    distill_loss = F.mse_loss(denoised_latents.float(),teacher_latents.float(), reduction='mean')
                    loss = distill_loss

                # [2] record vlb loss
                if args.vlb_loss:
                    text_file = os.path.join(pruned_layer_folder, f'vlb_loss.txt')
                else:
                    text_file = os.path.join(pruned_layer_folder, f'distill_loss.txt')
                with open(text_file, 'a') as f:
                    f.write(f'{loss.item()} \n')


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
    parser.add_argument('--vlb_loss', action="store_true")
    args = parser.parse_args()
    main(args)

