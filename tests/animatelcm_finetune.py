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
from diffusers import DDPMScheduler, DDIMScheduler
from safetensors.torch import save_file
from safetensors import safe_open
import numpy as np
from prun.utils.diffusion_misc import *
from prun.ode_solver import DDIMSolver
import torch.nn.functional as F
from prun.eval.VBench.vbench.motion_smoothness import MotionSmoothness
from prun.utils.clip_score import ClipScorer
from prun.data import DistillWebVid10M
from torch.utils.data import RandomSampler
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from einops import rearrange
import wandb
# evaluation metrics
def calculate_fvd(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(data1, model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)
    return fid_value

def huber_loss(pred, target, huber_c=0.001):
    loss = torch.sqrt((pred.float() - target.float()) ** 2 + huber_c ** 2) - huber_c
    return loss.mean()

def search_space_policy(original_state_dict, layers) :
    all_state_dicts = []
    for layer in layers :
        state_dict = original_state_dict.copy()
        for key in original_state_dict.keys() :
            if layer in key :
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
                                'up_blocks.0.motion_modules.0', 'up_blocks.0.motion_modules.1','up_blocks.0.motion_modules.2',
                                'up_blocks.1.motion_modules.0', 'up_blocks.1.motion_modules.1','up_blocks.1.motion_modules.2',
                                'up_blocks.2.motion_modules.0', 'up_blocks.2.motion_modules.1','up_blocks.2.motion_modules.2',
                                'up_blocks.3.motion_modules.0', 'up_blocks.3.motion_modules.1','up_blocks.3.motion_modules.2',]

def main(args):

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], )
    is_main_process = accelerator.is_main_process
    if is_main_process:
        run = wandb.init(project='animate_finetuning', name=args.wandb_run_name, group="DDP", )

    weight_dtype = torch.float32

    logger.info(f'\n step 3. saving dir')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f'\n step 5. preparing pruning')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=test_adapter,
                                                    torch_dtype=weight_dtype)
    test_pipe.scheduler = LCMScheduler.from_config(test_pipe.scheduler.config, beta_schedule="linear")
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()
    test_unet = test_pipe.unet  # is there lora ?

    logger.info(f'\n step 5. preparing teacher')
    device = accelerator.device
    noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", beta_schedule=args.beta_schedule, )
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(device, dtype=weight_dtype)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), ddim_timesteps=args.num_ddim_timesteps, )
    solver = solver.to(device)

    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=teacher_adapter, torch_dtype=weight_dtype)
    teacher_pipe.scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config, beta_schedule="linear")
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])
    teacher_pipe.enable_vae_slicing()
    teacher_pipe.enable_model_cpu_offload()
    teacher_pipe.to(accelerator.device)

    teacher_unet = teacher_pipe.unet
    vae = teacher_pipe.vae
    text_encoder = teacher_pipe.text_encoder
    tokenizer = teacher_pipe.tokenizer

    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    teacher_unet = teacher_unet.to(accelerator.device, dtype=weight_dtype)

    logger.info(f' (5.1) pruning')
    total_block_num = 21
    teacher_architecture = [i for i in range(total_block_num)]
    using_block_num = len(args.architecture)
    prun_arch = [x for x in teacher_architecture if x not in args.architecture]
    pruned_blocks = [total_blocks[i] for i in prun_arch]

    trainable_params = []
    for name, param in test_unet.named_parameters():
        prun = [block for block in pruned_blocks if block in name]
        if len(prun) > 0:
            param.data = torch.zeros_like(param.data) # set zero and not training
            param.requires_grad = False
        else:
            if 'motion' in name: # all motion module  training
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
    test_pipe.unet = test_unet
    test_pipe.to(accelerator.device)

    logger.info(f'\n step 6. preparing dataset (and loader)')
    csv.field_size_limit(sys.maxsize)
    args.csv_path = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample_filtered.csv'
    args.video_folder = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample_filtered'
    #args.csv_path = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv'
    #args.video_folder = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample'
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, drop_last=True)

    logger.info(f'\n step 7. set optimizer')
    optimizer = torch.optim.AdamW(trainable_params,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )

    logger.info(f'\n step 8. lr scheduler')
    lr_scheduler = get_scheduler(args.lr_scheduler,
                                 optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps,
                                 num_training_steps=args.max_train_steps)

    logger.info(f'\n step 9. prepare')
    test_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(test_unet,
                                                                               optimizer,
                                                                               train_dataloader,
                                                                               lr_scheduler)

    logger.info(f'\n step 10. saving argument')
    argument_file = os.path.join(output_dir, "args.json")
    with open(argument_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f'\n step 11. Train')
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")  # 10
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")  # 1
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Len train_dataloader = {len(train_dataloader)}")
    global_step = 0

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )
    global_step = 0

    for epoch in range(args.first_epoch, args.num_train_epochs):

        test_unet.train()
        teacher_unet.eval()

        """
        for step, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            with torch.no_grad():
                pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
                video_length = pixel_values.shape[1]
                pixel_value = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(accelerator.device, dtype=weight_dtype)

                # move module to device
                latents = vae.encode(pixel_value).latent_dist.sample()  # here problem ...
                latents = rearrange(latents, "(b f) c h w -> b c f h w",
                                    f=video_length)  # [batch, channel, frame, height, width]
                latents = latents * 0.18215  # always main process ?

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # [2] time embedding
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                # index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
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
                prompt_ids = tokenizer(batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                # encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
                if noise_scheduler.config.prediction_type == "epsilon": # here ... !
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # --------------------------------------------------------------------------------------------------- #
                teacher_unet = teacher_unet.to(noisy_model_input.device)
                teacher_output = teacher_unet(noisy_model_input, timesteps, encoder_hidden_states).sample
                teacher_pred_x_0 = get_predicted_original_sample(teacher_output,start_timesteps,
                                                                 noisy_model_input,"epsilon",alpha_schedule,sigma_schedule, )
                teacher_model_pred = c_skip_start * noisy_model_input + c_out_start * teacher_pred_x_0  # consistency

            student_output = test_unet(noisy_model_input, timesteps, encoder_hidden_states).sample  # [1,4,16,h,w]
            student_pred_x_0 = get_predicted_original_sample(student_output,start_timesteps,
                                                             noisy_model_input,"epsilon",alpha_schedule,sigma_schedule, )  # [1,4,16,h,w]
            student_model_pred = c_skip_start.to(student_pred_x_0.device) * noisy_model_input.to(student_pred_x_0.device) + c_out_start.to(student_pred_x_0.device) * student_pred_x_0  # consistency

            # [1.1] vlb loss
            # check weather all training is right or not
            vlb_loss = F.mse_loss(student_output.float(), target.float(), reduction='mean')

            # [1.2] distil loss (not use)
            if args.loss_type == "l2":
                distill_loss = F.mse_loss(student_model_pred.float(), teacher_model_pred.float(), reduction="mean")
            elif args.loss_type == "huber":
                distill_loss = huber_loss(student_model_pred, teacher_model_pred, args.huber_c)

            # [3] KL Divergence Loss
            
            kl_loss = kl_loss_fn(teacher_output, student_output)
            mu_t = teacher_model_pred.mean(dim=(1, 2))
            mu_s = student_model_pred.mean(dim=(1, 2))
            mu_loss = F.mse_loss(mu_t.to(mu_s.device).float(), mu_s.float(), reduction="mean")
            
            # [4]
            
            def sigmoid_t(x, binarize=False):
                result = 1 / (1 + torch.exp(-1 * 5 * x))
                if binarize:
                    result = (result > 0.5).float()
                return result
            
            frame_num = student_pred_x_0.shape[2]
            video_sim = 0.0
            for i in range(frame_num):
                frame_feature = student_pred_x_0[:,:,i,:,:]
                if i == 0:
                    first_image_feature = frame_feature
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_feature, frame_feature).mean()) # similar to former
                    sim_fir = max(0.0, F.cosine_similarity(first_image_feature, frame_feature).mean())  # similar to former
                    cur_sim = (sim_pre + sim_fir) / 2 # smaller than 1
                    video_sim += cur_sim
                former_image_feature = frame_feature
            sim_per_image = video_sim / (frame_num - 1)
            sim_loss = 1-sim_per_image

            if args.do_vlb_loss:
                loss = vlb_loss.mean()
            if args.do_distill_loss :
                loss = distill_loss.mean()
            
            if args.do_kl_loss:
                loss = kl_loss.mean()
            
            if args.frame_similarity_loss:
                loss = loss + sim_loss.mean()

            accelerator.backward(loss)

            # [6] optimizer step
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            accelerator.wait_for_everyone()

            if is_main_process:
                # loss = vlb_loss + use_ratio_loss * args.use_ratio_loss_scale + mu_loss * args.mu_loss_scale
                wandb.log({" vlb_loss": vlb_loss.item()})
                if args.frame_similarity_loss:
                    wandb.log({" sim_loss": sim_loss.item()})
        """
        accelerator.wait_for_everyone()
        if is_main_process:
            model_folder = os.path.join(output_dir, "pruned_model")
            os.makedirs(model_folder, exist_ok=True)
            test_unet = accelerator.unwrap_model(test_unet)
            state_dict = {"epoch": epoch,
                          "global_step": global_step,
                          "state_dict": test_unet.state_dict(),
                          }
            torch.save(state_dict, os.path.join(model_folder, f"checkpoint-epoch-{epoch + 1}.ckpt"))
            logging.info(f"Saved state to {model_folder} (global_step: {global_step})")

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
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--checkpointing_epochs', type=int, default=5)
    parser.add_argument('--checkpointing_steps', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--mixed_precision_training', action='store_true')
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
    parser.add_argument('--max_train_steps', type=int, default=30000)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="Num timesteps for DDIM sampling",
    )
    parser.add_argument("--beta_schedule",
                        default="scaled_linear",
                        type=str,
                        help="The schedule to use for the beta values.", )

    parser.add_argument('--frame_similarity_loss', action='store_true')
    parser.add_argument("--loss_type",type=str,default="huber",choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",)
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument("--wandb_run_name", type=str)
    args = parser.parse_args()
    main(args)


