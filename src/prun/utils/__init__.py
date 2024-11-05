import os
import argparse
import ast
import torchvision
from einops import rearrange
import numpy as np
import imageio
import torch
def arg_as_list(s) :
    v = ast.literal_eval(s)
    if type(v) is not list :
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)
