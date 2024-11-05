from copy import deepcopy
from einops import rearrange
from prun.data import DistillWebVid10M
from torch.utils.data import RandomSampler
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
import argparse
import functools
import gc
import itertools
import json
import logging
import math
import os, sys
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union
import csv, json
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
import webdataset as wds
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from braceexpand import braceexpand
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torch.utils.data import default_collate
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    LCMScheduler,
    StableDiffusionPipeline,UNetMotionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.training_utils import resolve_interpolation_mode
#from compression.prune_sd.calflops import calculate_flops
from copy import deepcopy
from prun.attn.controller import MotionControl
from prun.attn.masactrl_utils import register_motion_editor
from prun.utils.model_call import call_base_pipe
## loading tiny model w
MAX_SEQ_LENGTH = 77
import wandb
logger = get_logger(__name__)


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext) :param lcase: convert suffixes to
    lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {"__key__": prefix, "__url__": filesample["__url__"]}
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples

def log_validation(vae, unet, args, accelerator, weight_dtype, step, name="target"):
    logger.info("Running validation... ")

    test_pipe.unet = accelerator.unwrap_model(test_unet)
    test_pipe = test_pipe.to(accelerator.device)
    test_pipe.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "A yellow taxi cab sitting below tall buildings",
        "The shiny motorcycle has been put on display",
        "a professional photograph of an astronaut riding a horse, 8k",
        "muscular warrior fighting giant snake in the lake"
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            images = test_pipe(
                prompt=prompt,
                num_inference_steps=6,
                num_images_per_prompt=1,
                generator=generator,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})



# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.Tensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
    c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


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


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")




# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


def main(args):

    total_blocks_dot = ['down_blocks.0.motion_modules.0', 'down_blocks.0.motion_modules.1',
                    'down_blocks.1.motion_modules.0', 'down_blocks.1.motion_modules.1',
                    'down_blocks.2.motion_modules.0', 'down_blocks.2.motion_modules.1',
                    'down_blocks.3.motion_modules.0', 'down_blocks.3.motion_modules.1',
                    "mid_block.motion_modules.0",
                    'up_blocks.0.motion_modules.0', 'up_blocks.0.motion_modules.1', 'up_blocks.0.motion_modules.2',
                    'up_blocks.1.motion_modules.0', 'up_blocks.1.motion_modules.1', 'up_blocks.1.motion_modules.2',
                    'up_blocks.2.motion_modules.0', 'up_blocks.2.motion_modules.1', 'up_blocks.2.motion_modules.2',
                    'up_blocks.3.motion_modules.0', 'up_blocks.3.motion_modules.1', 'up_blocks.3.motion_modules.2', ]
    total_blocks = ['down_blocks_0_motion_modules_0', 'down_blocks_0_motion_modules_1',
                    'down_blocks_1_motion_modules_0', 'down_blocks_1_motion_modules_1',
                    'down_blocks_2_motion_modules_0', 'down_blocks_2_motion_modules_1',
                    'down_blocks_3_motion_modules_0', 'down_blocks_3_motion_modules_1',
                    "mid_block_motion_modules_0",
                    'up_blocks_0_motion_modules_0', 'up_blocks_0_motion_modules_1', 'up_blocks_0_motion_modules_2',
                    'up_blocks_1_motion_modules_0', 'up_blocks_1_motion_modules_1', 'up_blocks_1_motion_modules_2',
                    'up_blocks_2_motion_modules_0', 'up_blocks_2_motion_modules_1', 'up_blocks_2_motion_modules_2',
                    'up_blocks_3_motion_modules_0', 'up_blocks_3_motion_modules_1', 'up_blocks_3_motion_modules_2', ]

    print(f' step 1. logger and seed')
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              mixed_precision=args.mixed_precision,
                              log_with=args.report_to,
                              project_config=accelerator_project_config,
                              split_batches=True,)
    is_main_process = accelerator.is_local_main_process
    if is_main_process:
        run = wandb.init(project='ondevice-video',name=args.wandb_run_name,group="DDP", )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    print(f' step 2. Create the noise scheduler and the desired noise schedule.')
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler", revision=args.teacher_revision)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(), timesteps=noise_scheduler.config.num_train_timesteps, ddim_timesteps=args.num_ddim_timesteps,)

    print(f' step 3. get model')
    print(f' (3.1) teacher model')
    teacher_pipe = call_base_pipe()
    teacher_pipe.to(accelerator.device)
    teacher_unet = teacher_pipe.unet

    print(f' (3.2) sub models')
    tokenizer = teacher_pipe.tokenizer
    text_encoder = teacher_pipe.text_encoder
    text_encoder.requires_grad_(False)
    vae = teacher_pipe.vae
    vae.requires_grad_(False)

    print(f' (3.3) create online pruning unet ')
    test_pipe = call_base_pipe()
    test_unet = test_pipe.unet

    # teacher register
    teacher_motion_controller = MotionControl(is_teacher=True, train=True, frame_num=16, skip_layers=pruned_blocks, )
    register_motion_editor(teacher_unet, teacher_motion_controller)
    teacher_unet.requires_grad_(False)
    teacher_pipe.unet = teacher_unet

    trainable_params = []
    for name, param in test_unet.named_parameters():
        prun = [block for block in pruned_blocks_dot if block in name]
        if len(prun) > 0:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    test_unet.train()
    test_pipe.unet = test_unet
    test_pipe.to(accelerator.device)

    print(f' (2.3) target student U-Net')
    target_unet = deepcopy(test_unet)
    target_unet.eval()
    target_unet.requires_grad_(False)

    print(f' step 4. weight and device')
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16 # here ...
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print(f' final weight_dtype : {weight_dtype}')
    exit()
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    print(f' (4.1) mode lto device')
    vae.to(accelerator.device)
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    target_unet.to(accelerator.device)
    teacher_unet.to(accelerator.device)
    if args.cast_teacher_unet:
        teacher_unet.to(dtype=weight_dtype)
    alpha_schedule = alpha_schedule.to(accelerator.device)
    sigma_schedule = sigma_schedule.to(accelerator.device)
    solver = solver.to(accelerator.device)

    # 10. Handle saving and loading of checkpoints
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:

                target_unet.save_pretrained(os.path.join(output_dir, "unet_target"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, "unet_target"))
            target_unet.load_state_dict(load_model.state_dict())
            target_unet.to(accelerator.device)
            del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 11. Enable optimizations
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 12. Optimizer creation
    optimizer = optimizer_class(trainable_params,
                                lr=args.learning_rate,
                                betas=(args.adam_beta1, args.adam_beta2),
                                weight_decay=args.adam_weight_decay,
                                eps=args.adam_epsilon,)

    # 13. Dataset creation and data processing
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}

    ###########
    logger.info(f'\n step 6. preparing dataset (and loader)')
    csv.field_size_limit(sys.maxsize)
    # args.csv_path = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv'
    # args.video_folder = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample'
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
    ###########
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    # 14. LR Scheduler creation
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.per_gpu_batch_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 15. Prepare for training
    # Prepare everything with our `accelerator`.
    test_unet, optimizer, lr_scheduler = accelerator.prepare(test_unet, optimizer, lr_scheduler)
    # adding this argument to the model so that we can use it in the forward pass
    if isinstance(test_unet, torch.nn.parallel.DistributedDataParallel):
        test_unet.find_unused_parameters = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    uncond_input_ids = tokenizer(
        [""] * args.per_gpu_batch_size, return_tensors="pt", padding="max_length", max_length=77
    ).input_ids.to(accelerator.device)
    uncond_prompt_embeds = text_encoder(uncond_input_ids)[0]

    # 16. Train!
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # layerwise_hidden_dict
    for epoch in range(first_epoch, args.num_train_epochs):
        
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(test_unet):

                # 1. Load and process the image and text conditioning
                pixel_values = batch["pixel_values"]  # [batch, frame, channel, height, width]
                video_length = pixel_values.shape[1]
                pixel_value = rearrange(pixel_values, "b f c h w -> (b f) c h w").to(accelerator.device,
                                                                                     dtype=weight_dtype)
                # move module to device
                latents = vae.encode(pixel_value).latent_dist.sample()  # here problem ...
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)  # [batch, channel, frame, height, width]
                latents = latents * 0.18215  # always main process ?
                latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                text = batch['text']
                encoded_text = compute_embeddings_fn(text)

                # 2. Sample a random timestep for each image t_n from the ODE solver timesteps without bias.
                # For the DDIM solver, the timestep schedule is [T - 1, T - k - 1, T - 2 * k - 1, ...]
                topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 3. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps, timestep_scaling=args.timestep_scaling_factor)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps, timestep_scaling=args.timestep_scaling_factor)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

                # 4. Sample noise from the prior and add it to the latents according to the noise magnitude at each
                # timestep (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noise = torch.randn_like(latents)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                # 5. Sample a random guidance scale w from U[w_min, w_max] and embed it
                w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                time_cond_proj_dim = args.unet_time_cond_proj_dim
                w_embedding = guidance_scale_embedding(w,
                                                       embedding_dim=time_cond_proj_dim)
                w = w.reshape(bsz, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)

                # 6. Prepare prompt embeds and unet_added_conditions
                prompt_embeds = encoded_text.pop("prompt_embeds")

                # 7. Get online LCM prediction on z_{t_{n + k}} (noisy_model_input), w, c, t_{n + k} (start_timesteps)
                # [1] unet
                noise_pred = test_unet(
                    noisy_model_input,
                    start_timesteps,
                    # i have to make "time_cond_proj_dim"
                    # timestep_condition ......................................................
                    # timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds.float(),
                    added_cond_kwargs=encoded_text,
                ).sample
                student_layerwise_hidden_dict = student_motion_controller.layerwise_hidden_dict

                pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                # 8. Compute the conditional and unconditional teacher model predictions to get CFG estimates of the
                # predicted noise eps_0 and predicted original sample x_0, then run the ODE solver using these
                # estimates to predict the data point in the augmented PF-ODE trajectory corresponding to the next ODE
                # solver timestep.
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type)
                    with autocast_ctx:
                        # 1. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and conditional embedding c
                        cond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                        ).sample
                        teacher_layerwise_hidden_dict = teacher_motion_controller.layerwise_hidden_dict


                        cond_pred_x0 = get_predicted_original_sample(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        cond_pred_noise = get_predicted_noise(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 2. Get teacher model prediction on noisy_model_input z_{t_{n + k}} and unconditional embedding 0
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input.to(weight_dtype),
                            start_timesteps,
                            encoder_hidden_states=uncond_prompt_embeds.to(weight_dtype),
                        ).sample
                        uncond_pred_x0 = get_predicted_original_sample(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                        uncond_pred_noise = get_predicted_noise(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 3. Calculate the CFG estimate of x_0 (pred_x0) and eps_0 (pred_noise)
                        # Note that this uses the LCM paper's CFG formulation rather than the Imagen CFG formulation
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        pred_noise = cond_pred_noise + w * (cond_pred_noise - uncond_pred_noise)
                        # 4. Run one step of the ODE solver to estimate the next point x_prev on the
                        # augmented PF-ODE trajectory (solving backward in time)
                        # Note that the DDIM step depends on both the predicted x_0 and source noise eps_0.
                        x_prev = solver.ddim_step(pred_x0, pred_noise, index)


                # 9. Get target LCM prediction on x_prev, w, c, t_n (timesteps)
                # target unet predict on next timestep 
                with torch.no_grad():
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

                    with autocast_ctx:
                        target_noise_pred = target_unet(
                            x_prev.float(),
                            timesteps,
                            #timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds.float(),
                        ).sample
                    pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0

                # 10. Calculate loss
                # [10.1] LCM Loss
                if args.loss_type == "l2":
                    lcm_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_type == "huber":
                    lcm_loss = torch.mean(torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c ** 2) - args.huber_c)

                # [10.2] feature matching loss
                feature_loss = 0
                layer_num = len(teacher_layerwise_hidden_dict.keys())
                for layer in teacher_layerwise_hidden_dict.keys():
                    teacher_feature = teacher_layerwise_hidden_dict[layer]
                    student_feature = student_layerwise_hidden_dict[layer]
                    feature_loss = F.mse_loss(teacher_feature, student_feature, reduction="mean")
                    feature_loss += feature_loss
                feature_loss = feature_loss / layer_num

                # [10.3] total loss
                total_loss = lcm_loss + args.feature_matching_loss_weight * feature_loss
                # 11. Backpropagate on the online student model (`unet`)
                accelerator.backward(total_loss)
                wandb.log({"lcm_loss": lcm_loss.detach().item(),
                           "feature_loss": feature_loss.detach().item(),
                           "total_loss": total_loss.detach().item()}, step=global_step)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(test_unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 12. Make EMA update to target student model parameters (`target_unet`)
                update_ema(target_unet.parameters(), test_unet.parameters(), args.ema_decay)
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    #if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_folder = os.path.join(args.output_dir, f"pruned_model")
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, f"checkpoint-{global_step}")


                    unet_path = os.path.join(save_folder, f"unet")
                    os.makedirs(unet_path, exist_ok=True)
                    unet_to_save = accelerator.unwrap_model(test_unet).to('cpu')  # CPU로 이동
                    torch.save(unet_to_save.state_dict(), os.path.join(unet_path, f"checkpoint-{global_step}.pth"))

                    targt_unet_path = os.path.join(save_folder, f"unet_target")
                    os.makedirs(targt_unet_path, exist_ok=True)
                    target_unet_to_save = accelerator.unwrap_model(target_unet).to('cpu')  # CPU로 이동
                    torch.save(target_unet_to_save.state_dict(), os.path.join(targt_unet_path, f"checkpoint-{global_step}.pth"))

                    # validatio code

                    #if global_step % args.validation_steps == 0:
                    #    log_validation(vae, target_unet, args, accelerator, weight_dtype, global_step, "target")
                    #    log_validation(vae, unet, args, accelerator, weight_dtype, global_step, "online")

        #logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        #progress_bar.set_postfix(**logs)
        #accelerator.log(logs, step=global_step)
        #if global_step >= args.max_train_steps:
        #    break
        
        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            test_unet = accelerator.unwrap_model(test_unet)
            unet.save_pretrained(os.path.join(args.output_dir, "unet"))

            target_unet = accelerator.unwrap_model(target_unet)
            target_unet.save_pretrained(os.path.join(args.output_dir, "unet_target"))

            if args.push_to_hub:
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # ----------Wadb----------
    parser.add_argument("--wandb_run_name",type=str,)
    # ----------Model Checkpoint Loading Arguments----------


    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--teacher_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM teacher model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained LDM model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tiny_student_model",
        type=str,
        default=None,
        required=False,
        help="Path of the pruned bk sdm model.",
    )
    # ----------Training Arguments----------
    # ----General Training Arguments----
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lcm-xl-distilled",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # ----Logging----
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    # ----Checkpointing----
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    # ----Image Processing----
    parser.add_argument(
        "--train_shards_path_or_url",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--interpolation_type",
        type=str,
        default="bilinear",
        help=(
            "The interpolation function used when resizing images to the desired resolution. Choose between `bilinear`,"
            " `bicubic`, `box`, `nearest`, `nearest_exact`, `hamming`, and `lanczos`."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    # ----Dataloader----
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # ----Batch Size and Training Steps----

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    # ----Learning Rate----
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # ----Optimizer (Adam)----
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # ----Diffusion Training Arguments----
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--num_ddim_timesteps",
        type=int,
        default=50,
        help="The number of timesteps to use for DDIM sampling.",
    )

    parser.add_argument(
        "--unet_time_cond_proj_dim",
        type=int,
        default=256,
        help=(
            "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=32,
        required=False,
        help=(
            "The batch size used when encoding (and decoding) images to latents (and vice versa) using the VAE."
            " Encoding or decoding the whole batch at once may run into OOM issues."
        ),
    )
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
    # ----Exponential Moving Average (EMA)----
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    # ----Mixed Precision----
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cast_teacher_unet",
        action="store_true",
        help="Whether to cast the teacher U-Net to the precision specified by `--mixed_precision`.",
    )
    # ----Training Optimizations----
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    # ----Distributed Training----
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # ----------Validation Arguments----------
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    # ----------Huggingface Hub Arguments-----------
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    # ----------Accelerate Arguments----------
    parser.add_argument("--num_frames",type=int,default=16,)
    parser.add_argument("--datavideo_size",type=int,default=512)
    parser.add_argument("--guidance_scale",type=float,default=1.5)
    parser.add_argument("--num_inference_step",type=int,default=6)
    # num_inference_step
    from prun.utils import arg_as_list
    parser.add_argument("--architecture",type = arg_as_list)
    parser.add_argument('--sample_n_frames', type=int, default=16)
    parser.add_argument('--per_gpu_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--video_folder", type=str)
    # ----------Loss----------
    parser.add_argument("--loss_type",type=str,default="l2",choices=["l2", "huber"],help="The type of loss to use for the LCD loss.",)
    parser.add_argument("--huber_c",type=float,default=0.001,
                        help="The huber loss parameter. Only used if `--loss_type=huber`.",)
    parser.add_argument("--feature_matching_loss_weight", type=float, default=1)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    args = parser.parse_args()
    main(args)