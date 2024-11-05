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


# evaluation metrics
def calculate_fvd(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(data1, model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)
    return fid_value


def search_space_policy(original_state_dict, layers):
    all_state_dicts = []
    for layer in layers:
        state_dict = original_state_dict.copy()
        for key in original_state_dict.keys():
            if layer in key:
                state_dict[key] = torch.zeros_like(original_state_dict[key])
        all_state_dicts.append(state_dict)
    return all_state_dicts


def evaluation(prompt, evaluation_pipe, controller, save_folder,
               n_prompt, num_frames, guidance_scale, num_inference_steps, h=512, weight_dtype=torch.float16, p=0):
    prompt = prompt.strip()
    with torch.no_grad():
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
    if controller is not None:
        controller.reset()
        frames = output.frames[0]
        export_to_video(frames, os.path.join(save_folder, f'pruned_{prompt}.mp4'))
    else:
        frames = output.frames[0]

        # export_to_video(frames, os.path.join(save_folder, f'teacher_{prompt}.mp4'))
        export_to_video(frames, os.path.join(save_folder, f'sample_{p}.mp4'))
    # save on txt file
    text_file = os.path.join(save_folder, 'sample.txt')
    with open(text_file, 'a') as f:
        f.write(f'sample_{p},sample_{p}.mp4,{prompt}\n')


total_blocks = ['down_blocks.0.motion_modules.0', 'down_blocks.0.motion_modules.1',
                'down_blocks.1.motion_modules.0', 'down_blocks.1.motion_modules.1',
                'down_blocks.2.motion_modules.0', 'down_blocks.2.motion_modules.1',
                'down_blocks.3.motion_modules.0', 'down_blocks.3.motion_modules.1',
                "mid_block.motion_modules.0",
                'up_blocks.0.motion_modules.0', 'up_blocks.0.motion_modules.1', 'up_blocks.0.motion_modules.2',
                'up_blocks.1.motion_modules.0', 'up_blocks.1.motion_modules.1', 'up_blocks.1.motion_modules.2',
                'up_blocks.2.motion_modules.0', 'up_blocks.2.motion_modules.1', 'up_blocks.2.motion_modules.2',
                'up_blocks.3.motion_modules.0', 'up_blocks.3.motion_modules.1', 'up_blocks.3.motion_modules.2', ]


def main(args):

    logger.info(f'\n step 1. make dataset')
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
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers,
                                                   drop_last=True)


    for step, batch in enumerate(train_dataloader):

        pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width] = [1, 16, 3, 512, 512]
        # [2] get dynamic score



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
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--do_save_attention_map', action='store_true')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoints')
    parser.add_argument('--do_first_frame_except', action='store_true')
    parser.add_argument('--teacher_student_interpolate', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--training_test_merging', action='store_true')
    parser.add_argument("--target_change_index", type=int, default=512)
    parser.add_argument('--add_qkv', action='store_true')
    parser.add_argument('--inference_only_front_motion_module', action='store_true')
    parser.add_argument('--inference_only_post_motion_module', action='store_true')
    parser.add_argument('--pruning_test', action='store_true')
    parser.add_argument('--do_use_same_latent', action='store_true')
    parser.add_argument('--do_teacher_train_mode', action='store_true')
    parser.add_argument('--control_dim_dix', type=int, default=0)
    parser.add_argument('--pruned_unorder', action='store_true')
    parser.add_argument('--self_attn_pruned', action='store_true')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--pruning_ratio_list', type=arg_as_list, default=[])
    parser.add_argument('--target_time', type=int, default=0)
    parser.add_argument("--architecture",
                        type=arg_as_list,
                        default=[0, 1, 2, 3, 4, 6, 7, 14, 16, 18])
    # [7]
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2)
    # [8] training
    parser.add_argument('--first_epoch', type=int, default=0)
    parser.add_argument('--do_vlb_loss', action='store_true')
    parser.add_argument('--do_distill_loss', action='store_true')
    parser.add_argument('--do_kl_loss', action='store_true')
    parser.add_argument('--datavideo_size', type=int, default=512)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--per_gpu_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr_warmup_steps', type=int, default=0)
    parser.add_argument('--lr_scheduler', type=str, default='constant')
    parser.add_argument('--max_train_steps', type=int, default=3000)
    parser.add_argument('--train_batch_size', type=int, default=1)

    args = parser.parse_args()
    main(args)


