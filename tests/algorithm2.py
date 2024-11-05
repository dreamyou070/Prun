import random
import numpy as np
import argparse


class GeneticAlgorithm:

    def __init__(self,
                 total_block_num=21,
                 select_num=10,
                 population_size=50,
                 generations=100,
                 select_k=3,
                 test_pipe=None,
                 teacher_pipe=None,
                 test_prompts=None):

        # [1]
        self.total_block_num = total_block_num  # 전체 요소 개수 (0~20)
        self.select_num = select_num  # 뽑을 개수 (10개)
        self.max_combinations = np.math.comb(total_block_num, select_num)  # 가능한 조합 수
        self.population_size = population_size  # 초기 후보군 크기 (50개)
        assert self.population_size < self.max_combinations, "Population size must be smaller than the number of all possible combinations"
        self.generations = generations  # 총 반복할 세대 수 (100세대)
        self.population = self._initialize_population()
        self.select_k = select_k

        # [2] block model related
        self.total_blocks = ['down_blocks_0_motion_modules_0', 'down_blocks_0_motion_modules_1',
                             'down_blocks_1_motion_modules_0', 'down_blocks_1_motion_modules_1',
                             'down_blocks_2_motion_modules_0', 'down_blocks_2_motion_modules_1',
                             'down_blocks_3_motion_modules_0', 'down_blocks_3_motion_modules_1',
                             "mid_block_motion_modules_0",
                             'up_blocks_0_motion_modules_0', 'up_blocks_0_motion_modules_1',
                             'up_blocks_0_motion_modules_2',
                             'up_blocks_1_motion_modules_0', 'up_blocks_1_motion_modules_1',
                             'up_blocks_1_motion_modules_2',
                             'up_blocks_2_motion_modules_0', 'up_blocks_2_motion_modules_1',
                             'up_blocks_2_motion_modules_2',
                             'up_blocks_3_motion_modules_0', 'up_blocks_3_motion_modules_1',
                             'up_blocks_3_motion_modules_2', ]
        self.test_pipe = test_pipe
        if self.test_pipe is not None:
            self.test_unet = self.test_pipe.unet
        self.teacher_pipe = teacher_pipe
        if self.teacher_pipe is not None:
            self.teacher_unet = self.teacher_pipe.unet
            self.original_state_dict = self.teacher_unet.state_dict()

        # [3]
        self.test_record = {}
        self.test_prompts = test_prompts


    def _initialize_population(self):
        # 초기 개체 집합 생성 (예: 랜덤으로 생성)
        return [self.random_individual() for _ in range(self.population_size)]

    def random_individual(self):
        # 개체를 랜덤으로 생성하는 함수
        return sorted(np.random.choice(range(self.total_block_num), size=self.select_num, replace=False).tolist())

    def fitness(self, individual):
        # 개체의 적합도를 평가하는 함수
        return np.sum(individual)  # 예: 개체의 합을 적합도로 사용

    def select_top_k(self, k):
        # 적합도가 높은 상위 k개 개체 선택
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        return sorted_population[:k]

    def crossover(self, parent1, parent2):
        # 두 부모 개체 간의 교차 연산
        point = np.random.randint(1, len(parent1) - 1)
        # 두 부모의 유전자를 결합
        child = np.concatenate((parent1[:point], parent2[point:]))
        # 중복을 제거하고 정렬
        child = np.unique(child)[:self.select_num].tolist()
        # 자식 개체의 길이가 select_num보다 작을 경우 중복을 허용하지 않으므로, 추가값을 랜덤으로 선택
        while len(child) < self.select_num:
            additional_value = np.random.randint(0, self.total_block_num - 1)
            if additional_value not in child:
                child.append(additional_value)
        return sorted(child)

    def mutate(self, individual):
        # 개체에 변이를 적용하는 함수
        while True:
            index = np.random.randint(len(individual))
            mutated_value = np.random.randint(0, self.total_block_num - 1)
            # 중복 방지: 변이된 값을 개체에 대입
            if mutated_value not in individual:
                individual[index] = mutated_value
                individual = sorted(individual)
                break
        return individual

    def model_eval(self, architecture):

        # [1] total_block 에서 architecture를 가져와서 모델을 생성하고 성능을 평가하는 함수
        teacher_architecture = [i for i in range(self.total_block_num)]
        prun_arch = [x for x in teacher_architecture if x not in architecture]
        pruned_blocks = [self.total_blocks[i]for i in prun_arch]
        test_state_dict = self.original_state_dict.copy()
        for layer_name in test_state_dict.keys():
            prun = [block for block in pruned_blocks if block in layer_name]
            if prun > 0 :
                print(f'layer_name : {layer_name} prun!')
                test_state_dict[layer_name] = torch.zeros_like(test_state_dict[layer_name])
        self.test_unet.load_state_dict(test_state_dict).to('cuda')
        self.test_pipe.unet = self.test_unet
        self.test_pipe.to('cuda')
        """
        for p, prompt in enumerate(self.test_prompts):
            prompt = prompt.strip()
            with torch.no_grad():
                output = selt.test_pipe(prompt=prompt,
                                         negative_prompt=n_prompt,
                                         num_frames=num_frames,
                                         guidance_scale=guidance_scale,
                                         num_inference_steps=num_inference_steps,
                                         height=h,
                                         width=h,
                                         generator=torch.Generator("cpu").manual_seed(args.seed),
                                         dtype=weight_dtype,
                                         do_use_same_latent=args.do_use_same_latent)
                frames = output.frames[0]
                # list and len is frames
        """
        # recover
        self.test_unet.load_state_dict(self.original_state_dict).to('cuda')
        self.test_pipe.unet = self.test_unet

    def model_inference(self, architecture):
        # architecture is list
        if tuple(architecture) in self.test_record.keys():
            # if return False, what should I do?
            return False

        self.test_record[tuple(architecture)] = {}
        self.test_record[tuple(architecture)]['score'] = self.model_eval(architecture)
        self.test_record[tuple(architecture)]['done'] = True
        return True


    def evolve(self) :
        # [1] initial population
        init_architecture = self.population[0]
        self.model_inference(init_architecture)


        for generation in range(self.generations):
            # top 3 개체 선택
            parents = self.select_top_k(self.select_k)
            print(f"Generation {generation}: {parents}")

            # 자손 생성
            next_population = []
            while len(next_population) < self.population_size:
                # 두 부모를 무작위로 선택
                parent1, parent2 = random.sample(parents, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)

            self.population = next_population


def main(args):

    check_min_version("0.10.0.dev0")
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO, )

    logger.info(f'\n step 2. set seed')
    torch.manual_seed(args.seed)

    logger.info(f'\n step 3. preparing accelerator')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], )
    weight_dtype = torch.float32

    logger.info(f'\n step 4. saving dir')
    outpath = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    logging_dir = os.path.join(outpath, "logs.txt")

    logger.info(f'\n step 5. preparing pruning')
    test_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe=weight_dtype)
    test_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=test_adapter, torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(test_pipe.scheduler.config,beta_schedule="linear")
    test_pipe.scheduler = noise_scheduler
    test_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    test_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    test_pipe.enable_vae_slicing()
    test_pipe.enable_model_cpu_offload()

    logger.info(f'\n step 6. preparing teacher model')
    teacher_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM",torch_dtpe=weight_dtype)
    teacher_pipe = AnimateDiffPipeline.from_pretrained(args.pretrained_model_path, motion_adapter=teacher_adapter, torch_dtype=weight_dtype)
    noise_scheduler = LCMScheduler.from_config(teacher_pipe.scheduler.config,beta_schedule="linear")
    teacher_pipe.scheduler = noise_scheduler
    teacher_pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    teacher_pipe.set_adapters(["lcm-lora"], [0.8])  # LCM
    teacher_pipe.enable_vae_slicing()
    teacher_pipe.enable_model_cpu_offload()
    teacher_pipe.to('cuda')

    logger.info(f'\n step 7. calibration dataset')
    csv.field_size_limit(sys.maxsize)
    args.csv_path = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample.csv'
    args.video_folder = r'/scratch2/dreamyou070/MyData/video/panda/test_sample_trimmed/sample_'
    train_dataset = DistillWebVid10M(csv_path=args.csv_path,
                                     video_folder=args.video_folder,
                                     sample_size=args.datavideo_size,
                                     sample_stride=4,
                                     sample_n_frames=args.sample_n_frames,
                                     is_image=False)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler,
                                                   batch_size=args.per_gpu_batch_size,
                                                   num_workers=args.num_workers, drop_last=True)

    custom_prompt_dir = '/home/dreamyou070/Prun/src/prun/configs/prompts.txt'
    with open(custom_prompt_dir, 'r') as f:
        prompts = f.readlines()

    print(f' step 10. make GeneticAlgorithm instance')
    unprun_block_num = args.unprun_block_num
    population_size = args.population_size
    generations = args.generations
    select_k = args.select_k
    ga_searcher = GeneticAlgorithm(total_block_num = 21,
                                   select_num = unprun_block_num,
                                   population_size = population_size,
                                   generations = generations,
                                   select_k = select_k,
                                   test_pipe = test_pipe,
                                   teacher_pipe = teacher_pipe,
                                   test_prompts = prompts)
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
    parser.add_argument('--num_frames', type=int, default=16)
    from prun.utils import arg_as_list

    parser.add_argument('--skip_layers', type=arg_as_list, default=[])
    parser.add_argument('--skip_layers_dot', type=arg_as_list, default=[])
    parser.add_argument('--vlb_weight', type=float, default=1.0)
    parser.add_argument('--distill_weight', type=float, default=1.0)
    parser.add_argument('--loss_feature_weight', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=1.5)
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
    parser.add_argument('--num_inference_step', type=int)
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--vbench_dir", type=str)
    parser.add_argument("--training_modules", type=str)
    parser.add_argument("--frame_num", type=int, default=16)
    parser.add_argument("--lora_training", action='store_true')
    parser.add_argument("--motion_base_folder", type=str)
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument('--teacher_motion_model_dir', type=str, default="wangfuyun/AnimateLCM")
    parser.add_argument('--do_save_attention_map', action='store_true')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoints')
    parser.add_argument('--do_first_frame_except', action='store_true')
    parser.add_argument('--teacher_student_interpolate', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--training_test_merging', action='store_true')
    parser.add_argument("--target_change_index", type=int, default=512)
    parser.add_argument('--add_qkv', action='store_true')
    parser.add_argument('--inference_only_front_motion_module', action='store_true')
    parser.add_argument('--inference_only_post_motion_module', action='store_true')
    parser.add_argument('--pruning_test', action='store_true')

    parser.add_argument('--do_use_same_latent', action='store_true')
    parser.add_argument('--do_teacher_train_mode', action='store_true')
    parser.add_argument('--control_dim_dix', type=int, default=0)
    parser.add_argument('--pruned_unorder', action='store_true')
    parser.add_argument('--self_attn_pruned', action='store_true')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--pruning_ratio_list', type=arg_as_list, default=[])
    parser.add_argument('--target_time', type=int, default=0)
    # [10] Genetic Algorithm
    parser.add_argument('--unprun_block_num', type=int, default=10)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--select_k", type=int, default=3)

    args = parser.parse_args()
    main(args)
