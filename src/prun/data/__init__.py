import os, io, csv, math, random
import numpy as np
from decord import VideoReader
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class DistillWebVid10M(Dataset):
    def __init__(self,
                 csv_path,
                 video_folder,
                 sample_size=256,
                 sample_stride=4,
                 sample_n_frames=16,
                 is_image=False,
                 do_masking_loss = False,):
        print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)

        # self.pixel_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #    transforms.Resize(sample_size[0]),
        #    transforms.CenterCrop(sample_size),
        #    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),])
        
        self.pixel_transforms = transforms.Compose([transforms.Resize((sample_size[0],sample_size[0])),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),])
        
        self.do_masking_loss = do_masking_loss
        if self.do_masking_loss :
            mask_res = int(sample_size[0]/8)
            self.mask_transforms = transforms.Compose([transforms.Resize((mask_res, mask_res)),
                                                       transforms.CenterCrop((mask_res, mask_res))])


    def get_batch(self, idx):

        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        #video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_dir = os.path.join(self.video_folder, page_dir)
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1) # min(16,15*5)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.do_masking_loss :
            base_dir, sample_folder = os.path.split(video_dir)
            base_dir = os.path.split(base_dir)[0]
            mask_folder = os.path.join(base_dir, 'mask')
            mask_dir = os.path.join(mask_folder, f"{videoid}.mp4")
            mask_render = VideoReader(mask_dir)
            mask_values = torch.from_numpy(mask_render.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
            mask_values = mask_values / 255.
            del mask_render
        
        if self.is_image:
            pixel_values = pixel_values[0]
        # name
        if not self.do_masking_loss :
            return pixel_values, name
        else:
            return pixel_values, mask_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                if not self.do_masking_loss :
                    pixel_values, name = self.get_batch(idx)
                else:
                    pixel_values, mask_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        if not self.do_masking_loss :
            sample = dict(pixel_values=pixel_values,
                          text=name)
        else:
            mask_values = self.mask_transforms(mask_values)
            sample = dict(pixel_values=pixel_values, mask_values=mask_values, text=name)
        
        return sample
