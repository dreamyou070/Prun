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
from diffusers.pipelines import StableDiffusionPipeline
from diffusers import DDPMScheduler, DDIMScheduler


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

    logger.info(f'\n step 3. saving dir')
    n_prompt = "bad quality, worse quality, low resolution"
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_step

    logger.info(f'\n step 4. T2I Model')
    # make stable diffusion model
    test_pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_path,  torch_dtype=weight_dtype)
    test_pipe.scheduler = DDIMScheduler.from_config(test_pipe.scheduler.config)
    test_pipe.to(accelerator.device)

    logger.info(f'\n step 5. T2V Model')
    t2v_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    t2v_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=t2v_adapter,torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(t2v_pipe.scheduler.config, beta_schedule="linear")
    t2v_pipe.scheduler = noise_scheduler
    t2v_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",adapter_name="lcm-lora")
    t2v_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    t2v_pipe.enable_vae_slicing()
    t2v_pipe.enable_model_cpu_offload()

    logger.info(f'\n step 6. clip model')
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



    logger.info(f'\n step 6. reference prompts')
    custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        custom_prompts = f.readlines()


    logger.info(f'\n step 7. pruning')
    base_out_dir = args.output_dir
    os.makedirs(base_out_dir, exist_ok=True)

    logger.info(f'\n step 8. inference with prun model')
    for p, prompt in enumerate(custom_prompts):

        image = test_pipe(prompt=prompt,
                           negative_prompt=n_prompt,
                           guidance_scale=7.5,
                           num_inference_steps=30,
                           height=512,
                           width=512,
                           generator=torch.Generator("cpu").manual_seed(args.seed),
                           dtype=weight_dtype)[0][0] # pillow image
        image.save(os.path.join(base_out_dir, f'image_based_{prompt}.png'))
        # [2] clip feature extraction
        inputs = clip_processor(images=image, return_tensors="pt")

        image_features = clip_model.get_image_features(**inputs) # [1,768]
        # how to get image features
        # [77,768]

        print(f' image_features = {image_features}')
        # [3] t2v video generation
        video = t2v_pipe(prompt_embeds=image_features.unsqueeze(0), # [
                         negative_prompt=n_prompt,
                         guidance_scale=1.5,
                         num_inference_steps=6,
                         height=512,
                         width=512,
                         generator=torch.Generator("cpu").manual_seed(args.seed),
                         dtype=weight_dtype).frames[0]
        export_to_video(video, os.path.join(base_out_dir, f'image_based_{prompt}.mp4'))
        break

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
    args = parser.parse_args()
    main(args)


