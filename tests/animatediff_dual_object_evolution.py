import os
import numpy as np
import torch
import random
from ViCLIP.viclip import ViCLIP
from ViCLIP.simple_tokenizer import SimpleTokenizer
# 성능 지표를 하기 위한 후보군들
dimensions = ["subject_consistency", "background_consistency",'overall_consistency',
              "aesthetic_quality", "imaging_quality",
              "temporal_flickering", "motion_smoothness", "dynamic_degree"]

# ViCLIP = ViCLIP (Video Contrastive Language-Image Pretraining)
#

class GeneticAlgorithm:

    def __init__(self, total_blocks, teacher_pipe, original_state_dict, outpath, logging_dir):
        self.total_blocks = total_blocks
        self.teacher_pipe = teacher_pipe
        self.original_state_dict = original_state_dict
        self.outpath = outpath
        self.logging_dir = logging_dir
        self.test_record = {}
        self.total_block_num = len(total_blocks)
        self.pareto_front = []  # Pareto front를 저장할 리스트

        # [1] iamge quality

        # [3] video clip
        self.viclip_tokenizer = SimpleTokenizer('home/dreamyou070/.cashe/ViCLIP/bpe_simple_vocab_16e6.txt.gz')
        self.viclip = ViCLIP(tokenizer=tokenizer, **submodules_list).to(device)

    def fitness(self, save_folder):
        image_quality_score = self.evaluate_image_quality(save_folder)
        motion_dynamics_score = self.evaluate_motion_dynamics(individual)
        overall_consistency_score = self.evaluate_overall_consistency(save_folder)
        return overall_consistency_score, image_quality_score, motion_dynamics_score  # 두 점수를 반환

    def evaluate_overall_consistency(self, save_folder):

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
        def consistency(prompt, video_path):
            images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
            images = image_transform(images)
            images = images.to(device)
            # [3.1] img feature
            clip_feat = get_vid_features(self.viclip, images.unsqueeze(0))
            # [3.2] text feature
            text_feat = get_text_features(self.viclip, prompt, tokenizer)
            # [3.3] get score
            score = float((clip_feat @ text_feat.T)[0][0].cpu())
            return score

        video_files = os.listdir(save_folder)
        scores = []
        for video_file in video_files:

            video_path = os.path.join(save_folder, video_file)
            prompt = os.path.splitext(video_file)[0]
            score = consistency(prompt, video_path)
            scores.append(score)
        # sum of all scores
        mean_score = sum(scores)/len(scores)

        return mean_score

    def evaluate_image_quality(self, save_folder):
        files = os.listdir(save_folder)









    def evaluate_motion_dynamics(self, architecture):
        return np.random.rand()  # 임시로 랜덤 값을 반환

    def evaluate_overall_consistency(self, clip_model, video_dict, tokenizer, device, sample="middle"):
        sim = []
        video_results = []
        image_transform = clip_transform(224)


        for info in tqdm(video_dict, disable=get_rank() > 0):
            # [1] get prompt
            query = info['prompt']
            # text = clip.tokenize([query]).to(device)

            video_list = info['video_list']
            for video_path in video_list:
                cur_video = []
                with torch.no_grad():
                    # [2.1] read video to frames
                    images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
                    images = image_transform(images)
                    images = images.to(device)
                    # image feature
                    clip_feat = get_vid_features(clip_model, images.unsqueeze(0))
                    # text feature
                    text_feat = get_text_features(clip_model, query, tokenizer)
                    # image and text similarity
                    logit_per_text = clip_feat @ text_feat.T
                    score_per_video = float(logit_per_text[0][0].cpu())
                    sim.append(score_per_video)
                    video_results.append({'video_path': video_path, 'video_results': score_per_video})
        avg_score = np.mean(sim)
        return avg_score, video_results

    def model_inference(self, architecture, is_teacher=False):

        if is_teacher:
            self.model_eval_and_save(pipeline=self.teacher_pipe, save_folder=self.teacher_video_path)

        else:
            if tuple(architecture) in self.test_record.keys():
                return False
            self.test_record[tuple(architecture)] = {}
            teacher_architecture = [i for i in range(self.total_block_num)]
            prun_arch = [x for x in teacher_architecture if x not in architecture]
            pruned_blocks = [self.total_blocks[i] for i in prun_arch]
            test_state_dict = self.original_state_dict.copy()
            for layer_name in test_state_dict.keys():
                prun = [block for block in pruned_blocks if block in layer_name]
                if len(prun) > 0:
                    test_state_dict[layer_name] = torch.zeros_like(test_state_dict[layer_name])
            self.test_unet.load_state_dict(test_state_dict)
            self.test_pipe.unet = self.test_unet
            self.test_pipe.to('cuda')

            folder_name = "-".join(map(str, architecture))
            save_folder = os.path.join(self.outpath, f'pruned_using_{folder_name}_architecture')
            self.model_eval_and_save(self.test_pipe, save_folder=save_folder)
            self.test_record[tuple(architecture)]["done"] = True

            overall_consistency, image_quality, motion_dynamics = self.fitness(save_folder)
            self.test_record[tuple(architecture)]["overall_consistency"] = overall_consistency
            self.test_record[tuple(architecture)]["image_quality"] = image_quality
            self.test_record[tuple(architecture)]["motion_dynamics"] = motion_dynamics

            # 파레토 프론트에 추가
            self.update_pareto_front(image_quality, motion_dynamics, architecture)

            # logging
            with open(self.logging_dir, 'a') as f:
                f.write(f"Architecture {architecture} : {self.test_record[tuple(architecture)]}\n")

            self.test_unet.load_state_dict(self.original_state_dict)
            self.test_unet.to('cuda')
            self.test_pipe.unet = self.test_unet
            self.test_pipe.to('cuda')

    def update_pareto_front(self, image_quality, motion_dynamics, architecture):
        """
        Update the Pareto front based on the current architecture's scores.
        """
        new_solution = (image_quality, motion_dynamics, architecture)
        self.pareto_front.append(new_solution)

        # 파레토 최적 해를 찾기 위해 기존 해와 비교
        self.pareto_front = self.non_dominated_sort(self.pareto_front)

    def non_dominated_sort(self, solutions):
        """
        Returns the non-dominated solutions from a list of solutions.
        """
        non_dominated = []
        for s in solutions:
            if not any(self.dominates(other, s) for other in non_dominated):
                non_dominated = [other for other in non_dominated if not self.dominates(s, other)]
                non_dominated.append(s)
        return non_dominated

    def dominates(self, s1, s2):
        """
        Checks if solution s1 dominates solution s2.
        """
        return (s1[0] >= s2[0] and s1[1] > s2[1]) or (s1[0] > s2[0] and s1[1] >= s2[1])

    def model_eval_and_save(self, pipeline, save_folder):
        pass

    def get_fvd_score(self, generated_folder, reference_folder):
        return np.random.rand()


def generate_initial_population(population_size, num_blocks_to_select):
    """
    Generate initial population of architectures.
    number of blocks to select is the number of blocks to select for each architecture.
    """
    total_blocks = ["block1", "block2", "block3"]
    population = []
    for _ in range(population_size):
        architecture = random.sample(total_blocks, k=num_blocks_to_select)
        population.append(architecture)
    return population


def main() :

    # [1] 예제 사용
    teacher_pipe = None  # Teacher pipeline을 설정해야 함
    original_state_dict = {}  # 원래의 상태 딕셔너리 설정 필요
    outpath = "./output"
    logging_dir = "./logs.txt"
    population_size = 10  # 초기 개체 수
    generations = 5  # 세대 수

    # [2] GeneticAlgorithm 객체 생성
    ga = GeneticAlgorithm(total_blocks, teacher_pipe, original_state_dict, outpath, logging_dir)

    # [3] 초기 인구 생성
    population = generate_initial_population(total_blocks, population_size)

    # [4] 세대 수만큼 반복
    for generation in range(generations): # 0,1,2,3,4
        print(f"Generation {generation + 1}")
        for architecture in population:
            # population
            ga.model_inference(architecture)  # 각 아키텍처 평가

    # 최종 파레토 프론트 출력
    print("Final Pareto Front:")
    for solution in ga.pareto_front:
        print(f"Architecture: {solution[2]}, Image Quality: {solution[0]}, Motion Dynamics: {solution[1]}")

if __name__ == "__main__":
    main()