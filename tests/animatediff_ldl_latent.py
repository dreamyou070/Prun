import random
import numpy as np
import argparse
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
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video
import csv, json
from safetensors.torch import save_file
from safetensors import safe_open
import numpy as np
from torchvision import models, transforms
from safetensors.torch import load_file
import cv2
import math
import scipy
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
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
import numpy as np
from torchvision import models, transforms
import ast
import pandas as pd

def arg_as_list(s) :
    v = ast.literal_eval(s)
    if type(v) is not list :
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v

class GeneticAlgorithm:

    def __init__(self,
                 total_block_num=21,
                 select_num=10,
                 population_size=50,
                 mutation_num = 9,
                 crossover_num = 10,
                 generations=100,
                 select_k=3,
                 test_pipe=None,
                 teacher_pipe=None,
                 test_prompts=None,
                 weight_dtype=None,
                 outpath=None,
                 max_no_change = 3,
                 init_architecture =  [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18], #  [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18]
                 target_block_num = 10,
                 max_prompt = 10,
                 total_blocks = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # [1]
        self.total_block_num = total_block_num  # 전체 요소 개수 (0~20)
        self.select_num = select_num  # 뽑을 개수 (10개)
        self.crossover_num = crossover_num
        self.select_k = select_k
        self.max_combinations = math.comb(total_block_num, select_num)  # 가능한 조합 수
        self.population_size = population_size  # 초기 후보군 크기 (50개)
        assert self.population_size < self.max_combinations, "Population size must be smaller than the number of all possible combinations"
        self.mutation_num = mutation_num
        self.generations = generations  # 총 반복할 세대 수 (100세대)
        self.test_record = {}
        # [2] block model related
        
        self.total_blocks = total_blocks
        self.test_pipe = test_pipe
        if self.test_pipe is not None:
            self.test_unet = self.test_pipe.unet
        self.teacher_pipe = teacher_pipe
        if self.teacher_pipe is not None:
            self.teacher_unet = self.teacher_pipe.unet
            self.original_state_dict = self.teacher_unet.state_dict()
        self.weight_dtype = weight_dtype
        self.init_architecture = init_architecture #[0,1,2,3,4,13,14,16,17,18]
        print(f'init_architecture : {self.init_architecture}')
        print(f'len of init_architecture : {len(self.init_architecture)}')
        self.target_block_num = target_block_num
        print(f'target_block_num : {self.target_block_num}')
        assert len(self.init_architecture) == self.target_block_num, "Initial architecture must have the same number of blocks as the target block number"

        # [3] test
        self.test_record = {}
        self.test_prompts = test_prompts
        #self.inception_model = self._load_inceptionv3_model().to(self.device)
        self.top_arch = {}

        # [4] path
        self.outpath = outpath
        os.makedirs(outpath, exist_ok=True)
        self.teacher_video_path = os.path.join(outpath, "teacher_folder")
        os.makedirs(self.teacher_video_path, exist_ok=True)
        self.teacher_feature_folder = os.path.join(outpath, "teacher_feature")
        os.makedirs(self.teacher_feature_folder, exist_ok=True)
        self.logging_dir = os.path.join(outpath, "logs.txt")
        self.max_no_change = max_no_change
        self.candidates = []
        self.max_prompt = max_prompt

    def _load_inceptionv3_model(self):
        model = models.inception_v3(pretrained=True)
        model.fc = nn.Identity()  # Remove the final classification layer
        model.eval()
        return model

    def fitness(self, individual):
        # 개체의 적합도를 평가하는 함수
        return np.sum(individual)  # 예: 개체의 합을 적합도로 사용

    def update_top_k(self, select_num, sorting_key, reverse=False):
        """ sorting test_record by fvd score and update top_arch again """
        # 만약 reverse 가 False 이면 작은수부터 정렬하게 된다.
        # sorted_by_key 는 fvd 가 작은 순으로 뽑게 되어 가장 teacher 과 가까운 값이 뽑히게 된다.
        sorted_by_key = sorted(self.test_record.items(), key=sorting_key, reverse=reverse)
        # 가장 teacher 와 유사한 값을 보이는 구조가 뽑혔다.
        self.top_arch = list(dict(sorted_by_key[:select_num]).keys())

    def model_eval_and_save(self, pipeline, save_folder):

        parent, folder_name = os.path.split(save_folder)
        
        # [0] save video
        video_folder = os.path.join(save_folder, 'sample')
        os.makedirs(video_folder, exist_ok = True)

        # [1] video save
        tsv_file = os.path.join(save_folder, f'pseudo_fvd.tsv')
        with open(tsv_file, 'a') as f:
            f.write(f'prompt\tmu\tstd\n')
        
        for p, prompt in enumerate(self.test_prompts):
            save_prompt = f'prompt_{p}'
            if p < self.max_prompt :
               output = pipeline(prompt=prompt,
                                  negative_prompt=args.n_prompt,
                                  num_frames=args.frame_num,
                                  guidance_scale=args.guidance_scale,
                                  num_inference_steps=args.num_inference_steps,
                                  height=args.h, width=args.h,
                                  generator=torch.Generator("cpu").manual_seed(args.seed),
                                  dtype=self.weight_dtype,
                                  output_type = "latent").frames
               mu, std = torch.mean(output).item(), torch.std(output).item() # how to get this std ?
               # [0] save sample
               with torch.no_grad() :
                   video_tensor = pipeline.decode_latents(output, 16) # [1,3,16,512,512]
                   video = pipeline.video_processor.postprocess_video(video=video_tensor, output_type='pil')[0] # list of 16 pillows
                   export_to_video(video, os.path.join(video_folder, f'{save_prompt}.mp4'))
               
                # [1] save statistics
               if p == self.max_prompt - 1:
                   with open(tsv_file, 'a') as f:
                       f.write(f'{save_prompt}\t{mu}\t{std}')
               else:
                   with open(tsv_file, 'a') as f:
                       f.write(f'{save_prompt}\t{mu}\t{std}\n')

    def get_fvd_score(self, folder1, folder2):
        prun_csv = os.path.join(folder1, 'pseudo_fvd.tsv')
        prun_df = pd.read_csv(prun_csv, delimiter="\t")

        teacher_csv = os.path.join(folder2, 'pseudo_fvd.tsv')
        teacher_df = pd.read_csv(teacher_csv, delimiter="\t")

        fvd_scores = []
        for i in range(len(prun_df)):
            mu_diff = (prun_df['mu'][i] - teacher_df['mu'][i]) ** 2
            std_diff = (prun_df['std'][i] - teacher_df['std'][i]) ** 2
            print(f'mu_diff : {mu_diff}, std_diff : {std_diff}')
            if type(mu_diff) == torch.Tensor:
                mu_diff = mu_diff.item()
            if type(std_diff) == torch.Tensor:
                std_diff = std_diff.item()
            score = mu_diff + std_diff
            fvd_scores.append(score)
        # get mean of fvd scores
        fvd_arr = np.array(fvd_scores)
        fvd = np.mean(fvd_arr)
        prun_model_score_dir = os.path.join(folder1, 'fvd_score.txt')
        with open(prun_model_score_dir, 'w') as f:
            f.write(f'{fvd}')
        return fvd

    def model_inference(self, architecture, is_teacher=False, ):

        if is_teacher:
            self.model_eval_and_save(pipeline = self.teacher_pipe, save_folder = self.teacher_video_path)
        else :
            # only if
            if tuple(architecture) in self.test_record.keys():
                return False
            # make pipeline
            # [1] total_block 에서 architecture를 가져와서 모델을 생성하고 성능을 평가하는 함수
            self.test_record[tuple(architecture)] = {}

            # [1] make pruning model
            teacher_architecture = [i for i in range(self.total_block_num)]
            prun_arch = [x for x in teacher_architecture if x not in architecture]
            pruned_blocks = [self.total_blocks[i] for i in prun_arch] # pruned_blocks
            test_state_dict = self.original_state_dict.copy()
            for layer_name in test_state_dict.keys():
                prun = [block for block in pruned_blocks if block in layer_name]
                if len(prun) > 0: # this will be pruned ...
                    test_state_dict[layer_name] = torch.zeros_like(test_state_dict[layer_name]) # make it zero ...
            self.test_unet.load_state_dict(test_state_dict)
            #.to('cuda')
            self.test_pipe.unet = self.test_unet
            self.test_pipe.to('cuda')

            # [2] set save folder

            folder_name = ""
            for i in architecture: # using ?
                folder_name += f'{i}-'
            save_folder = os.path.join(self.outpath, f'pruned_using_{folder_name}architecture')

            # [3] model eval and save
            self.model_eval_and_save(self.test_pipe, save_folder=save_folder) # after tht

            # [4] record
            self.test_record[tuple(architecture)]["done"] = True
            self.test_record[tuple(architecture)]["fvd"] = self.get_fvd_score(save_folder, self.teacher_video_path)
            # logging
            with open(self.logging_dir, 'a') as f:
                f.write(f"Architecture {architecture} : {self.test_record[tuple(architecture)]}\n")

            # recover
            self.test_unet.load_state_dict(self.original_state_dict)
            self.test_unet.to('cuda')
            self.test_pipe.unet = self.test_unet
            self.test_pipe.to('cuda')
            #self.test_record[tuple(architecture)] = {}
            #self.test_record[tuple(architecture)]['score'] = self.model_eval(architecture)
            #self.test_record[tuple(architecture)]['done'] = True
    """
    def extract_inception_features(self, frames):
        preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
        frames = [preprocess(frame) for frame in frames]
        frames = torch.stack(frames).to(self.device)
        with torch.no_grad():
            features = self.inception_model.to(self.device)(frames)  # frames, all_dimension (framewise)
        return features.cpu().numpy()
    """
    def read_video(self, video_path, max_frames=64):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (299, 299))
            frames.append(frame)  # list of frame
        cap.release()
        frames = np.array(frames)  # [frame_num=16, h, w, c=3]
        return frames

    def set_reference(self,) :
        self.model_inference(architecture=None, is_teacher=True)

    def mutate_architecture(self, architecture, num, m_prob=.1):
        # architecture = [1,2,3]
        mutated_architectures = []
        max_iters = num * 10
        mutate_candidates = sorted([i for i in range(self.total_block_num) if i not in architecture]) # [5,6,7,8,9,10,11,12,19,20]

        def mutate_with_prob(architecture):
            for i in range(len(architecture)):
                if np.random.random_sample() < m_prob:
                    mutated_value = np.random.choice(mutate_candidates) # mutate value
                    # numpy to integer
                    if int(mutated_value) not in architecture:
                        architecture[i] = int(mutated_value)
                        architecture = sorted(architecture)
                        # it cannot be same
                        # or it can be different not one time but more
            return sorted(architecture)

        while len(mutated_architectures) < num and max_iters > 0 :
            max_iters -= 1
            mutated_architecture = mutate_with_prob(architecture)
            if tuple(mutated_architecture) not in self.candidates:
                mutated_architectures.append(mutated_architecture)
        return mutated_architectures

    def cross_architecture(self, num):

        def crossover(parent1, parent2):
            new_cand = []
            for i in range(len(parent1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(parent1[i])
                else:
                    new_cand.append(parent2[i])  # it can be same ...
            new_cand = sorted(np.unique(new_cand).tolist())
            while len(new_cand) < len(parent1):
                additional_value = np.random.randint(0, self.total_block_num - 1)
                if additional_value not in new_cand:
                    new_cand.append(additional_value)
            return sorted(new_cand)

        children = []
        for i in range(num):
            # present generation self.candidates ...
            parent1, parent2 = random.sample(self.candidates, 2)
            child = crossover(parent1, parent2)
            if tuple(child) not in self.candidates:
                children.append(child)
        return children

    def evolve(self) :

        # total record = self.test_record
        # [1.1] initial population : 1
        init_architecture = self.init_architecture
        self.candidates.append(tuple(init_architecture)) #
        ################################################################################################################################################
        """
        self.test_record[tuple(init_architecture)] = {}
        self.test_record[tuple(init_architecture)]["done"] = False
        folder_name = ""
        for i in init_architecture:  # using ?
            folder_name += f'{i}-'
        save_folder = os.path.join(self.outpath, f'pruned_using_{folder_name}architecture')
        self.test_record[tuple(architecture)]["fvd"] = self.get_fvd_score(save_folder, self.teacher_video_path)
        self.test_record[tuple(init_architecture)]["fvd"] = 0
        """
        ################################################################################################################################################
        self.model_inference(init_architecture, is_teacher=False) # self.test_record

        # [1.2] mutation : population_size = 10
        mutate_architecture = self.mutate_architecture(init_architecture, num = self.mutation_num) # mutation should be 19

        for i_m in mutate_architecture :
            self.candidates.append(tuple(i_m))
            self.model_inference(i_m, is_teacher=False)

        # [1.3] cross
        cross_num = self.population_size - len(self.candidates)
        children = self.cross_architecture(num = cross_num)
        for child in children:
            self.candidates.append(tuple(child))
            self.model_inference(child, is_teacher=False)

        # [1.4] update top_k
        # 가장 좋은 구조들이 뽑히게 된다.
        # model 의 folder 은
        self.update_top_k(select_num=self.select_k,sorting_key=lambda x: x[1]["fvd"], ) # self.top_arch = [(),()]
        print(f' [Generation 0] : {self.top_arch}')
        previous_top = self.top_arch

        # --------------------------------------------------------------------------------------------------
        g_i = 1
        no_change_count = 0
        for generation in range(self.generations):
            # [2.1] initial population
            self.candidates = self.top_arch # [[0,1,2,3,4,13,14,15,16,17,18], [0,1,2,3,4,13,14,15,16,17,18]]

            # [2.2] mutation : randomly select one
            random_choice_arch = random.choice(self.top_arch)
            mutate_architecture = self.mutate_architecture(list(random_choice_arch),
                                                           num=self.mutation_num)  # mutation should be 19
            for i_m in mutate_architecture:
                self.candidates.append(tuple(i_m))
                self.model_inference(i_m, is_teacher=False)

            # [2.3] crossover
            cross_num = self.population_size - len(self.candidates)
            children = self.cross_architecture(num=cross_num)
            for i_c in children:
                self.candidates.append(tuple(i_c))
                self.model_inference(i_c, is_teacher=False)

            # [2.4] update top_k
            self.update_top_k(select_num=self.select_k, sorting_key=lambda x: x[1]["fvd"])
            current_top = self.top_arch
            print(f' [Generation {g_i}] : {current_top}')
            if previous_top == current_top:
                no_change_count += 1  # 변동이 없으면 카운트 증가

            if no_change_count > self.max_no_change :
                break
            else :
                previous_top = current_top
                g_i += 1


def main(args):

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], )
    dtype = torch.float32
    weight_dtype = torch.float32

    logger.info(f'\n step 4. saving dir')
    outpath = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)


    logger.info(f'\n step 5. preparing pruning')
    device = "cuda"
    test_adapter = MotionAdapter().to(device, dtype)
    motion_path = '/home/jovyan/pretrained/AnimateDiff-Lightning/animatediff_lightning_4step_diffusers.safetensors'
    test_adapter.load_state_dict(load_file(motion_path, device=device))    
    model_path = '/home/jovyan/pretrained/epiCRealism'
    test_pipe = AnimateDiffPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    test_pipe.scheduler = EulerDiscreteScheduler.from_config(test_pipe.scheduler.config,
                                                             timestep_spacing="trailing", beta_schedule="linear")
    print(f' step 2. check blocks')
    unet = test_pipe.unet
    attention_modules, motion_modules = [], []
    def check_children(net, name):
        for net_name, sub_net in net.named_children():
            final_name = f"{name}{net_name}"
            if sub_net.__class__.__name__ == 'Transformer2DModel' :
                attention_modules.append(final_name)
            elif sub_net.__class__.__name__ == 'TransformerTemporalModel':
                motion_modules.append(final_name)
            if hasattr(net, 'children'):
                check_children(sub_net, f'{final_name}.')
    check_children(unet, "")

    print(f'len of attention_modules : {len(attention_modules)}') # 16 = 2/2/2/1/3/3/3
    print(f'len of motion_modules : {len(motion_modules)}') # 21
    total_control_blocks = attention_modules + motion_modules # 37
    print(f'total_control_blocks : {len(total_control_blocks)}')

    print(f' step 4. algorithmic search')
    init_sd = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15] # only erase 8th block
    init_motion = [0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18]
    init_motion = [i + len(attention_modules) for i in init_motion]
    init_architecture = init_sd + init_motion

    logger.info(f'\n step 6. preparing teacher model')
    teacher_adapter = MotionAdapter().to(device, dtype)
    motion_path = '/home/jovyan/pretrained/AnimateDiff-Lightning/animatediff_lightning_4step_diffusers.safetensors'
    teacher_adapter.load_state_dict(load_file(motion_path, device=device))    
    model_path = '/home/jovyan/pretrained/epiCRealism'
    teacher_pipe = AnimateDiffPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    teacher_pipe.scheduler = EulerDiscreteScheduler.from_config(teacher_pipe.scheduler.config,
                                                                timestep_spacing="trailing",
                                                                beta_schedule="linear")                                                             
    teacher_pipe.to('cuda')

    custom_prompt_dir = '/home/jovyan/Prun/src/prun/configs/animal_filtered_webvid10m_test.txt'
    with open(custom_prompt_dir, 'r') as f:
        prompts = f.readlines()

    logger.info(f'\n step 8. make GeneticAlgorithm instance')
    unprun_block_num = args.unprun_block_num
    population_size = args.population_size
    generations = args.generations
    select_k = args.select_k
    ga_searcher = GeneticAlgorithm(total_block_num = 21,
                                   select_num = unprun_block_num,
                                   population_size = population_size,
                                   mutation_num = args.mutation_num,
                                   crossover_num = args.crossover_num,
                                   generations = generations,
                                   select_k = select_k,
                                   test_pipe = test_pipe,
                                   teacher_pipe = teacher_pipe,
                                   test_prompts = prompts,
                                   outpath = outpath,
                                   weight_dtype = weight_dtype,
                                   max_no_change = args.max_no_change,
                                   init_architecture = init_architecture,
                                   target_block_num = len(init_architecture),
                                   max_prompt = args.max_prompt,
                                   total_blocks = total_control_blocks)

    print(f' step 9. evolution algorithm')
    ga_searcher.set_reference()

    print(f' step 10. generic evolution')
    ga_searcher.evolve()



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
    parser.add_argument('--n_prompt', type=str, default='bad video')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
    parser.add_argument('--num_inference_steps', type=int, default=6)
    parser.add_argument("--h", type=int, default=512)


    # [2]
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
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
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--vbench_dir", type=str)
    parser.add_argument("--training_modules", type=str)
    parser.add_argument("--frame_num", type=int, default=16)
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--target_time', type=int, default=0)
    # [10] Genetic Algorithm
    parser.add_argument('--unprun_block_num', type=int, default=10)
    parser.add_argument("--population_size", type=int, default=20)
    parser.add_argument("--mutation_num", type=int, default=9) #
    parser.add_argument("--crossover_num",type=int,default=10,)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--select_k", type=int, default=2)
    parser.add_argument("--max_no_change", type=int, default=3)
    parser.add_argument("--target_block_num", type=int, default=10)
    parser.add_argument("--init_architecture",
                        type=arg_as_list,
                        default=[0, 1, 2, 3, 4, 13, 14, 15, 16, 17, 18])
    parser.add_argument("--max_prompt", type=int, default=10)
    args = parser.parse_args()
    main(args)
