
import os
from fvd_calculate import calculate_fvd

def main() :
    experiment_dir='/home/dreamyou070/VideoDistill/experiment'
    base_folder='teacher_nontraining_panda_base_sample'
    teacher_folder=f"{experiment_dir}/{base_folder}/teacher_folder/eval_samples/teacher_mp4"

    student_base_folder='1_layer_skip'
    student_folder_base=f"{experiment_dir}/{student_base_folder}/eval_samples"

    folders=["student_epoch_001_mp4","student_epoch_002_mp4","student_epoch_003_mp4","student_epoch_004_mp4","student_epoch_005_mp4"]
    for folder in folders :
        student_folder = f"{student_folder_base}/{folder}"
        teacher_files = os.listdir(teacher_folder)
        for file in teacher_files :
            teacher_video_path = os.path.join(teacher_folder, file)
            student_video_path = os.path.join(student_folder, file)
            fvd_score = calculate_fvd(teacher_video_path, student_video_path)
            #break

if __name__ == '__main__' :
    main()