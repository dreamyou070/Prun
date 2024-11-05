import random
import os
import torch
import numpy as np
import argparse
from prun.utils import arg_as_list
from diffusers.utils import check_min_version
import logging
from prun.score_fn.dynamic_degree import compute_dynamic_degree
from prun.score_fn.motion_smoothness import compute_motion_smoothness
from prun.score_fn.subject_consistency import compute_subject_consistency
from accelerate import Accelerator
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif, export_to_video

class MultiObjectiveGeneticAlgorithm:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1, total_block_num=21,
                 test_pipe=None, test_prompts=None, max_prompt=10, weight_dtype=torch.float32,
                 output_dir='output'):
        # 클래스 초기화: 인구 수, 세대 수, 돌연변이 비율, 블록 수 설정
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.total_block_num = total_block_num
        self.total_blocks = ['down_blocks.0.motion_modules.0', 'down_blocks.0.motion_modules.1',
                             'down_blocks.1.motion_modules.0', 'down_blocks.1.motion_modules.1',
                             'down_blocks.2.motion_modules.0', 'down_blocks.2.motion_modules.1',
                             'down_blocks.3.motion_modules.0', 'down_blocks.3.motion_modules.1',
                             "mid_block.motion_modules.0",
                             'up_blocks.0.motion_modules.0', 'up_blocks.0.motion_modules.1',
                             'up_blocks.0.motion_modules.2',
                             'up_blocks.1.motion_modules.0', 'up_blocks.1.motion_modules.1',
                             'up_blocks.1.motion_modules.2',
                             'up_blocks.2.motion_modules.0', 'up_blocks.2.motion_modules.1',
                             'up_blocks.2.motion_modules.2',
                             'up_blocks.3.motion_modules.0', 'up_blocks.3.motion_modules.1',
                             'up_blocks.3.motion_modules.2', ]
        self.target_block_num = 10
        # [2] pipe
        self.test_pipe = test_pipe
        self.unet = self.test_pipe.unet
        self.weight_dtype = weight_dtype
        self.original_state_dict = self.unet.state_dict()

        # [3] evaluation
        self.test_prompts = test_prompts
        self.max_prompt = max_prompt
        args.n_prompt = 'bad'
        args.frame_num = 16
        args.guidance_scale = 1.5
        args.num_inference_steps = 6
        args.h = 512
        self.seed = 0
        self.weight_dtype = torch.float32
        # 초기 개체군 생성 (50개), 각 아키텍처는 블록의 리스트로 표현
        self.population = self.generate_initial_population()
        # [2] save folder
        self.base_folder = output_dir
        os.makedirs(self.base_folder, exist_ok=True)

    def generate_initial_population(self):
        # 초기 개체군을 생성하는 메서드
        population = []
        for _ in range(self.population_size):
            # 임의로 10 개 블록을 샘플링 하고 sort 해서 아키텍처 생성
            architecture = sorted(random.sample(range(self.total_block_num), k=self.target_block_num))
            population.append(architecture)  # 생성된 아키텍처를 개체군에 추가
        return population
    def model_inference_and_save(self, architecture, folder):

        # [1.1] make pruning model
        if architecture is not None:
            teacher_arch = [i for i in range(self.total_block_num)]
            prun_arch = [x for x in teacher_arch if x not in architecture]
            pruned_blocks = [self.total_blocks[i] for i in prun_arch] # pruned_blocks
            test_state_dict = self.original_state_dict.copy()
            # [1.2] make pruned model
            for layer_name in test_state_dict.keys():
                prun = [block for block in pruned_blocks if block in layer_name]
                if len(prun) > 0: # this will be pruned ...
                    test_state_dict[layer_name] = torch.zeros_like(test_state_dict[layer_name]) # make it zero ...
            self.unet.load_state_dict(test_state_dict)
        self.test_pipe.unet = self.unet
        self.test_pipe.to('cuda')

        # [2] inference
        for p, prompt in enumerate(self.test_prompts):
            save_prompt = prompt.strip().replace(' ', '-')

            if p < self.max_prompt:
                output = self.test_pipe(prompt=prompt,
                                  negative_prompt=args.n_prompt,
                                  num_frames=args.frame_num,
                                  guidance_scale=args.guidance_scale,
                                  num_inference_steps=args.num_inference_steps,
                                  height=args.h, width=args.h,
                                  generator=torch.Generator("cpu").manual_seed(args.seed),
                                  dtype=self.weight_dtype, )
                frames = output.frames[0]  # 16 len of pillows
                p = prompt.strip()
                export_to_video(frames, os.path.join(folder, f'{save_prompt}.mp4'))
        # [3] recover
        self.unet.load_state_dict(self.original_state_dict)
        self.unet.to('cuda')
        self.test_pipe.unet = self.unet
        self.test_pipe.to('cuda')
        self.device = 'cuda'

    def evaluate_motion_dynamics(self, folder):
        # [1] load video
        video_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        dynamic_all_results, dynamic_mean = compute_dynamic_degree(self.device, video_list)
        return dynamic_mean
    def evaluate_motion_smoothness(self, folder):

        video_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        motion_smoothness_all_results, motion_smoothness_mean = compute_motion_smoothness(self.device, video_list)
        return motion_smoothness_mean

    def evaluate_subject_consistency(self, folder):
        video_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        subject_consistency_all_results, subject_consistency_mean = compute_subject_consistency(self.device, video_list)
        return subject_consistency_mean

    def evaluate_architecture(self, architecture):

        if architecture is None:
            # teacher model
            architecture_folder = os.path.join(self.base_folder, f'teacher')

        else :
            # 해당 구조를 이용해서 모델을 만들고 비디오를 생성하기
            folder_name = ""
            for i in architecture:  # using ?
                folder_name += f'{i}-'
            architecture_folder = os.path.join(self.base_folder, f'pruned_using_{folder_name}architecture')

        os.makedirs(architecture_folder, exist_ok=True)
        self.model_inference_and_save(architecture, folder = architecture_folder)
        dynamic_mean = self.evaluate_motion_dynamics(architecture_folder)  # 모션 부드러움 점수 생성
        motion_smooth_mean = self.evaluate_motion_smoothness(architecture_folder)
        subjecyt_consistency = self.evaluate_subject_consistency(architecture_folder)

        return dynamic_mean, motion_smooth_mean, subjecyt_consistency

    def sort_population(self):
        # 개체군을 적합도에 따라 정렬하는 메서드
        return sorted(self.population, key=lambda arch: self.evaluate_architecture(arch), reverse=True)

    def crossover(self, parent1, parent2):
        # 두 부모 아키텍처를 받아 교차 연산 수행
        return parent1[:len(parent1)//2] + parent2[len(parent2)//2:]

    def mutate(self, architecture):
        # 주어진 아키텍처에 변이를 적용하는 메서드
        if random.random() < self.mutation_rate:  # 변이 확률 체크
            block_to_mutate = random.randint(0, self.num_blocks - 1)  # 변이할 블록 선택
            if block_to_mutate in architecture:
                architecture.remove(block_to_mutate)  # 블록 제거
            else:
                architecture.append(block_to_mutate)  # 블록 추가
        return architecture  # 변이가 적용된 아키텍처 반환

    def reference(self):

        self.evaluate_architecture(arch=None)
    def run(self):

        # 초기 개체군을 평가
        self.population = self.sort_population()  # 초기 개체군 정렬

        # 유전 알고리즘의 주요 실행 루프
        for generation in range(self.generations):
            print(f' [Generation] {generation + 1}')
            # 개체군 정렬
            sorted_population = self.sort_population()
            # 최상위 두 개체 선택
            next_population = sorted_population[:2]

            # 다음 세대의 개체군을 생성
            while len(next_population) < self.population_size:
                # 상위 10개 개체 중 두 개 선택 : 부모 세대
                parent1 = random.choice(sorted_population[:10])
                parent2 = random.choice(sorted_population[:10])
                # 교차 연산 수행 후 변이 적용
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)  # 자식 아키텍처를 다음 세대에 추가

            self.population = next_population  # 현재 세대를 다음 세대로 업데이트

        # 최상의 아키텍처와 그에 대한 적합도 메트릭 출력
        best_architecture = self.sort_population()[0]
        best_metrics = self.evaluate_architecture(best_architecture)
        print(f"Best architecture: {best_architecture}, Metrics: {best_metrics}")

def main(args) :

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    accelerator = Accelerator()
    weight_dtype = torch.float32

    logger.info(f'\n step 5. preparing pruning')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtpe=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=test_adapter,
                                                    torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(test_pipe.scheduler.config, beta_schedule="linear")
    test_pipe.scheduler = noise_scheduler
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                                adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()

    logger.info(f'\n step 6. genetic algorithm')
    ga = MultiObjectiveGeneticAlgorithm(total_block_num=21,
                                        test_pipe=test_pipe,
                                        test_prompts=args.prompt_file_dir,
                                        max_prompt= 1, #args.max_prompt,
                                        weight_dtype=weight_dtype,
                                        output_dir=args.output_dir,)
    ga.run()  # 유전 알고리즘 실행


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # [1]
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
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--pruning_ratio_list', type=arg_as_list, default=[])
    parser.add_argument('--target_time', type=int, default=0)
    # [10] Genetic Algorithm
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
    parser.add_argument("--pretrained_model_path", type=str, default='prompt.txt')
    # inference
    parser.add_argument("--prompt_file_dir", type=str, default='prompt.txt')
    args = parser.parse_args()
    main(args)