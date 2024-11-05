import numpy as np
import torch

features_1 = torch.randn((16,2048)).numpy()
mu1, sigma1 = np.mean(features_1, axis=0), np.cov(features_1, rowvar=False) # every feature mean and variance

print(f'mu1 = {mu1.shape}, sigma1 = {sigma1.shape}')    