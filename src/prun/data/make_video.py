from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import csv
import os
import clip
import torch

nsfw_nonnsfw_concepts = ["violent, death, blood, wounds, mutilation, injury, gore, graphic",
                         "nudity, naked, explicit, private parts, unclothed, bare, nude",
                         "pornography, explicit, sexual, adult, mature, x-rated, obscene",
                         "explicit, sexual, intercourse, graphic, adult, mature, obscene",
                         "child, minor, exploitation, inappropriate, sexual, abuse",
                         "solicitation, sexual, explicit, adult, services, proposition",
                         "violence, gore, violent, blood, wounds, injury, death",
                         "suicide, self-harm, self-injury, self-destructive, death, kill",
                         "harassment, bullying, cyberbullying, threat, intimidation, abuse",
                         "hate, discrimination, racism, bigotry, prejudice, intolerance",
                         "intolerance, discrimination, bigotry, prejudice, bias, hate",
                         "drugs, narcotics, controlled substances, illegal, abuse, misuse",
                         "alcohol, drinking, drunk, intoxication, abuse, underage",
                         "tobacco, smoking, cigarettes, nicotine, underage, addiction",
                         "weapons, guns, firearms, violence, illegal, dangerous",
                         "gambling, bet, wager, casino, risk, addiction",
                         "controversial, sensitive, divisive, polarizing, debate, conflict",]

def main(args):

    print(f' \n step 1. make Motion Base Pipeline with LCM Scheduler')
    adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter,
                                               torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                           adapter_name="lcm-lora")
    pipe.set_adapters(["lcm-lora"], [0.8])
    pipe.enable_vae_slicing()
    device = 'cuda'
    pipe.to(device)

    print(f' \n step 2. call nsfw model')
    model, preprocess = clip.load("ViT-B/32", device=device)
    # [2.1] Tokenize
    nsfw_concepts_text_tokens = clip.tokenize(nsfw_nonnsfw_concepts).to(device)
    with torch.no_grad():
        nsfw_text_features = model.encode_text(nsfw_concepts_text_tokens)

    print(f' \n step 2. save_base_dir')
    save_base_dir = f'../MyData/video/webvid_genvideo'
    os.makedirs(save_base_dir, exist_ok=True)
    sample_folder = os.path.join(save_base_dir, 'sample')
    os.makedirs(sample_folder, exist_ok=True)

    print(f' \n step 3. inference test')
    print(f' (3.1) prompt check')
    prompt_dir = f'./configs/training/filtered_captions_train_trimmed.txt'
    with open(prompt_dir, 'r') as f:
        prompts = f.readlines()
    print(f' (3.2) others check')
    num_frames = 16
    guidance_scale = 1.5
    inference_scale = 6
    n_prompt = "bad quality, worse quality, low resolution"
    seed = 0
    print(f' (3.3) check nsfw')
    safe_prompts = []
    for p, prompt in enumerate(prompts):
        with torch.no_grad():
            prompt_tokens = clip.tokenize(prompt).to(device)
            prompt_features = model.encode_text(prompt_tokens)
            # [2] Normalize
            nsfw_text_features /= nsfw_text_features.norm(dim=-1, keepdim=True)
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
            # [3] similarity and results
            similarity = (100.0 * prompt_features @ nsfw_text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            if values[0].item() < 0.5:  # pass
                safe_prompts.append(prompt)


    print(f' \n step 4. inference')
    elems = []
    header = ['videoid', 'page_dir', 'name']
    for p, prompt in enumerate(safe_prompts):
        save_p = str(p).zfill(7)
        output = pipe(prompt=prompt,
                      negative_prompt=n_prompt,
                      num_frames=num_frames,
                      guidance_scale=guidance_scale,
                      num_inference_steps=inference_scale,
                      generator=torch.Generator("cpu").manual_seed(seed), )
        frames = output.frames[0]
        export_to_video(frames, os.path.join(sample_folder, f'prompt_sample_{save_p}.mp4'))
        elem = [f'prompt_sample_{save_p}',
                os.path.join(sample_folder, f'prompt_sample_{save_p}.mp4'),
                prompt]
        elems.append(elem)
    csv_file = os.path.join(save_base_dir, f'webvid_genvideo_2_7.csv')
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(header)
        # write content
        writer.writerows(elems)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='t2v_inference')
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--is_teacher', action='store_true')
    parser.add_argument('--start_num', type=int, default=100)
    parser.add_argument('--end_num', type=int, default=140)
    args = parser.parse_args()
    main(args)