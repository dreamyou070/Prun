import os
import json
import numpy as np
import torch
import clip
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer


tokenizer = SimpleTokenizer('/home/dreamyou070/.cache/vbench/ViCLIP/bpe_simple_vocab_16e6.txt.gz')
viclip = ViCLIP(tokenizer= tokenizer,
                pretrain = '/home/dreamyou070/.cache/vbench/ViCLIP/ViClip-InternVid-10M-FLT.pth').to(device)

def compute_overall_consistency(prompt, video_path):
    def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
        if input_text in text_feature_dict:
            return text_feature_dict[input_text]
        text_template = f"{input_text}"
        with torch.no_grad():
            text_features = model.encode_text(text_template).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_feature_dict[input_text] = text_features
        return text_features

    def get_vid_features(model, input_frames):
        with torch.no_grad():
            clip_feat = model.encode_vision(input_frames, test=True).float()
            clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
        return clip_feat

    image_transform = clip_transform(224)
    # ------------------------------------------------------------------------------------------ #
    images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
    images = image_transform(images)
    images = images.to(device)
    # [3.1] img feature
    clip_feat = get_vid_features(viclip, images.unsqueeze(0))
    # [3.2] text feature
    text_feat = get_text_features(viclip, prompt, tokenizer)
    # [3.3] get score
    score = float((clip_feat @ text_feat.T)[0][0].cpu())
    return score

def main() :
    prompt = 'a video'
    video_path = '/scratch2/dreamyou070/Prun/result/1_teacher_calibration_data/sample/sample_0.mp4'
    score = compute_overall_consistency(prompt, video_path)
    print(score)

if __name__ == '__main__':
    main()