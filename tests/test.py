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
#from prun.eval.VBench.vbench.motion_smoothness import MotionSmoothness
#from prun.utils.clip_score import ClipScorer


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

    total_blocks_dot = ['down_blocks.0.motion_modules.0', 'down_blocks.0.motion_modules.1',
                        'down_blocks.1.motion_modules.0', 'down_blocks.1.motion_modules.1',
                        'down_blocks.2.motion_modules.0', 'down_blocks.2.motion_modules.1',
                        'down_blocks.3.motion_modules.0', 'down_blocks.3.motion_modules.1',
                        "mid_block.motion_modules.0",
                        'up_blocks.0.motion_modules.0', 'up_blocks.0.motion_modules.1', 'up_blocks.0.motion_modules.2',
                        'up_blocks.1.motion_modules.0', 'up_blocks.1.motion_modules.1', 'up_blocks.1.motion_modules.2',
                        'up_blocks.2.motion_modules.0', 'up_blocks.2.motion_modules.1', 'up_blocks.2.motion_modules.2',
                        'up_blocks.3.motion_modules.0', 'up_blocks.3.motion_modules.1',
                        'up_blocks.3.motion_modules.2', ]
    total_blocks = ['down_blocks_0_motion_modules_0', 'down_blocks_0_motion_modules_1',
                    'down_blocks_1_motion_modules_0', 'down_blocks_1_motion_modules_1',
                    'down_blocks_2_motion_modules_0', 'down_blocks_2_motion_modules_1',
                    'down_blocks_3_motion_modules_0', 'down_blocks_3_motion_modules_1',
                    "mid_block_motion_modules_0",
                    'up_blocks_0_motion_modules_0', 'up_blocks_0_motion_modules_1', 'up_blocks_0_motion_modules_2',
                    'up_blocks_1_motion_modules_0', 'up_blocks_1_motion_modules_1', 'up_blocks_1_motion_modules_2',
                    'up_blocks_2_motion_modules_0', 'up_blocks_2_motion_modules_1', 'up_blocks_2_motion_modules_2',
                    'up_blocks_3_motion_modules_0', 'up_blocks_3_motion_modules_1', 'up_blocks_3_motion_modules_2', ]

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO, )

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

    logger.info(f'\n step 5. preparing pruning')
    # check LCM Lora again
    """
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    test_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=test_adapter,torch_dtype=torch.float16)
    test_pipe.scheduler = LCMScheduler.from_config(test_pipe.scheduler.config, beta_schedule="linear")
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()
    test_unet = test_pipe.unet  # is there lora ?
    """
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=test_adapter,torch_dtype=weight_dtype)
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM",weight_name="AnimateLCM_sd15_t2v_lora.safetensors",adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()
    test_unet = test_pipe.unet
    total_block_num = 21
    # [1] original architecture = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    teacher_architecture = [i for i in range(total_block_num)]
    prun_arch = [x for x in teacher_architecture if x not in args.architecture]
    pruned_blocks = [total_blocks[i] for i in prun_arch]
    pruned_blocks_dot = [total_blocks_dot[i] for i in prun_arch]
    # make controller
    student_motion_controller = MotionControl(is_teacher=False,
                                  train=True,
                                  frame_num=16,
                                  skip_layers=pruned_blocks,)
    # change to simple adapter
    register_motion_editor(test_unet, student_motion_controller)
    if args.test_unet_dir is not None :
        tensors = {}
        with safe_open(args.test_unet_dir, framework="pt", device=0) as f:
            for k in f.keys():
                check_2 = [prun_block for prun_block in pruned_blocks_dot if prun_block in k]
                if len(check_2) > 0 :
                    # only bias and weight ...
                    print(f' input key = {k}')
                    tensors[k] = f.get_tensor(k)
                    tensors[k] = tensors[k] * 0

        # load state dict
        test_unet.load_state_dict(tensors, strict=False)

    test_unet = test_unet.to(accelerator.device, dtype=weight_dtype)
    test_pipe.unet = test_unet

    logger.info(f'\n step 6. load trained state dict (pruned)')
    custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        custom_prompts = f.readlines()

    logger.info(f'\n step 7. folder')
    base_out_dir = args.output_dir
    os.makedirs(base_out_dir, exist_ok = True)
    base_out_dir = os.path.join(base_out_dir, 'sample')
    os.makedirs(base_out_dir, exist_ok = True)
    base_out_dir = os.path.join(base_out_dir, 'checkpoint_1')
    os.makedirs(base_out_dir, exist_ok=True)



    # [1] smoothness model
    """
    config = '/home/dreamyou070/Prun/src/prun/eval/VBench/vbench/third_party/amt/cfgs/AMT-S.yaml'
    ckpt = '/home/dreamyou070/.cache/vbench/amt_model/amt-s.pth'
    smoothness_model = MotionSmoothness(config=config,ckpt=ckpt, device='cuda')
    # [2] clip model
    clip_scorer = ClipScorer(device = "cuda")
    
    # [3] aesthetic model
    from prun.utils.aesthetic_score import compute_aesthetic_quality
    """
    global_num = 0

    logger.info(f'\n step 8. inference with prun model')
    for p, prompt in enumerate(custom_prompts):
        if p < 50 :
            evaluation(prompt,
                       test_pipe,
                       None,
                       base_out_dir,
                       n_prompt,
                       num_frames,
                       guidance_scale,
                       num_inference_steps,
                       h=args.h,
                       weight_dtype=weight_dtype,
                       p=p)

        """
        # [2] scoring (teacher)
        if global_num == 0 :
            text_align_scores = []
            video_list = []
            files = os.listdir(teacher_out_dir)
            for file in files :
                file_path = os.path.join(teacher_out_dir, file)
                video_list.append(file_path)
                name = os.path.splitext(file)[0]
                prompt = name.replace('-0','')
                clip_score = clip_scorer.clip_score(prompt, [file_path])
                text_align_scores.append(clip_score)
            mean_ttscore = sum(text_align_scores) / len(text_align_scores)
            mean_t_aes_score, video_results = compute_aesthetic_quality(video_list, device = accelerator.device)
        
        # [2] scoring (pruned)
        total_smoothness_scores = []
        total_text_align_scores = []
        test_files = os.listdir(save_dir)

        pruned_video_list = []
        for file in test_files :
            if 'mp4' in file :
                path = os.path.join(save_dir, file)

                pruned_video_list.append(path)
                # [1] smoothness (here problem)
                score = smoothness_model.motion_score(path)
                total_smoothness_scores.append(score)
                # [2] clip score
                name = os.path.splitext(file)[0]
                name = name.replace('-0','')
                prompt = name.replace('pruned_','')
                clip_score = clip_scorer.clip_score(prompt,[path])
                total_text_align_scores.append(clip_score)
        mean_sm_score = sum(total_smoothness_scores) / len(total_smoothness_scores)
        mean_ta_score = sum(total_text_align_scores) / len(total_text_align_scores)
        mean_aes_score, _ = compute_aesthetic_quality(pruned_video_list, device = accelerator.device)
        print(f'mean text alignment score = {mean_ta_score}')

        # aesthetic score criteria
        score_text = os.path.join(save_dir, 'score.txt')
        with open(score_text, 'w') as f :
            f.write(f' motion smoothness score = {str(mean_sm_score)}')
            f.write(f'\n text align score = {str(mean_ta_score)}')
            f.write(f'\n aesthetic score = {str(mean_aes_score)}')
        """



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
    parser.add_argument('--test_unet_dir', type=str)
    parser.add_argument("--architecture",
                        type=arg_as_list,
                        default=[0, 1, 2, 3, 4, 6, 7, 14, 16, 18])
    args = parser.parse_args()
    main(args)


