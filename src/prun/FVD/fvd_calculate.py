"""
DISCLAIMER
This implementation is largely inspired by the implemenation from the StyleGAN-V repository
https://github.com/universome/stylegan-v
However, it is adapted to videos in main memory and simplified.
The authors of the StyleGAN-V repository verified the consistency of their PyTorch implementation with the original Tensorflow implementation.
The original implementation can be found here: https://github.com/google-research/google-research/tree/master/frechet_video_distance

The link used to download the pretrained feature extraction model was provided by the StyleGAN-V authors. I cannot garantuee it is still working.
"""

import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset
import torch
import hashlib
import os
import glob
import requests
import re
import html
import io
import uuid
import numpy as np
import cv2
import tensorflow as tf
from scipy.linalg import sqrtm

import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
from scipy.linalg import sqrtm
from torchvision import models, transforms


# Load pre-trained InceptionV3 model for feature extraction
def load_inceptionv3_model():

    model = models.inception_v3(pretrained=True)
                                #aux_logits=False)

    model.fc = nn.Identity()  # Remove the final classification layer
    model.eval()
    return model


# Read video and preprocess to extract frames
def read_video(video_path, max_frames=64):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frames.append(frame) # list of frame
    cap.release()
    frames = np.array(frames) # [frame_num=16, h, w, c=3]
    return frames


# Extract features using the InceptionV3 model
def extract_features(model, frames, device):

    # np to torch 
    preprocess = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),])
    frames = [preprocess(frame) for frame in frames] # proces
    
    # should be
    frames = torch.stack(frames).to(device)
    with torch.no_grad():
        features = model(frames) # frames=16, all_dimension=2048
    return features.cpu().numpy()


# Calculate the Frechet Distance
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


# Calculate FVD between two videos
def calculate_fvd(video_path_1, video_path_2):

    # [1] load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_inceptionv3_model().to(device)

    # [2] read video
    frames_1 = read_video(video_path_1) # [batch, frame_num, h, w, c=3], numpy array
    frames_2 = read_video(video_path_2)


    # [3] get features
    features_1 = extract_features(model, frames_1, device) # torch [16 frame, 2048 dim]
    features_2 = extract_features(model, frames_2, device)

    # [4] 
    mu1, sigma1 = np.mean(features_1, axis=0), np.cov(features_1, rowvar=False) # all dimension
    mu2, sigma2 = np.mean(features_2, axis=0), np.cov(features_2, rowvar=False)

    # [4] calculate fvd
    fvd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fvd