import numpy as np
from PIL import Image
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from transformers import CLIPProcessor, CLIPModel
import cv2

NUM_ASPECT=5
ROUND_DIGIT=3
MAX_LENGTH = 76

MAX_NUM_FRAMES=16

CLIP_POINT_LOW=0.27
CLIP_POINT_MID=0.31
CLIP_POINT_HIGH=0.35

X_CLIP_POINT_LOW=0.15
X_CLIP_POINT_MID=0.225
X_CLIP_POINT_HIGH=0.30

class ClipScorer:

    def __init__(self, device):
        self.model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")  # .to(device)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device

    def x_clip_score(self, text: str, frame_path_list: List[str], ):
        
        # should be list of frrames
        # how to get frames
        """
        def _read_video_frames(frame_paths, max_frames):
            
            total_frames = len(frame_paths)
            print(f'total_frames = {total_frames}')
            
            indices = np.linspace(0, total_frames - 1, num=max_frames).astype(int)
            
            # here problem ...
            selected_frames = [np.array(Image.open(frame_paths[i])) for i in indices]
            return np.stack(selected_frames)
        """
        # [1] get text feature        
        input_text = self.tokenizer([text], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt").to(self.device)
        text_feature = self.model.get_text_features(**input_text).flatten()

        # [2] get image feature
        #video = _read_video_frames(frame_path_list, MAX_NUM_FRAMES)
        video_path = frame_path_list[0]
        cap = cv2.VideoCapture(video_path)
        # 총 프레임 수 얻기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 16
        frames_to_extract = total_frames // 16
        frame_list = []
        for i in range(16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_to_extract)
            ret, frame = cap.read()
            frame_list.append(frame)
        video = np.stack(frame_list)
        input_video = self.processor(videos=list(video), return_tensors="pt").to(self.device)
        video_feature = self.model.get_video_features(**input_video).flatten().to

        # [3] get score (between text and video frames)
        cos_sim = F.cosine_similarity(text_feature.to('cpu'),video_feature.to('cpu'), dim=0).item()

        return cos_sim

    def clip_score(self, text: str, frame_path_list: List[str], ):
        device = self.clip_model.device
        input_t = self.clip_tokenizer(text=text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt", padding=True).to(device)
        cos_sim_list = []
        
        video_path = frame_path_list[0]
        cap = cv2.VideoCapture(video_path)
        # 총 프레임 수 얻기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 16
        frames_to_extract = total_frames // 16
        frame_list = []
        for i in range(16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frames_to_extract)
            ret, frame = cap.read()
            image = Image.fromarray(frame)
            input_f = self.clip_tokenizer(images=image, return_tensors="pt", padding=True).to(device)
            output_t = self.clip_model.get_text_features(**input_t).flatten()
            output_f = self.clip_model.get_image_features(**input_f).flatten()
            cos_sim = F.cosine_similarity(output_t, output_f, dim=0).item()  # torch score
            cos_sim_list.append(cos_sim)
        """
        for frame_path in frame_path_list:
            image = Image.open(frame_path)
            input_f = self.clip_tokenizer(images=image, return_tensors="pt", padding=True).to(device)
            output_t = self.clip_model.get_text_features(**input_t).flatten()
            output_f = self.clip_model.get_image_features(**input_f).flatten()
            cos_sim = F.cosine_similarity(output_t, output_f, dim=0).item()  # torch score
            cos_sim_list.append(cos_sim)
        """
        clip_score_avg = np.mean(cos_sim_list)

        return clip_score_avg