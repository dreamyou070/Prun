import os
import subprocess


def convert_mpg_to_mp4(input_file, output_file):

    ffmpeg_path = "ffmpeg.exe"
    command = [ffmpeg_path, "-version"]
    subprocess.run(command)
    print(r' run finish ! ')
    try:
        # FFmpeg 명령어를 리스트로 실행 (터미널에서 실행되는 것과 동일)
        print(f' input_file: {input_file}, -> output_file: {output_file}')
        command = ['ffmpeg', '-i', input_file, '-vcodec', 'libx264', '-acodec', 'aac', output_file]

        subprocess.run(command, check=True)
        print(f"변환 완료! {output_file}로 저장되었습니다.")
    except subprocess.CalledProcessError as e:
        print("명령어 실행 중 오류가 발생했습니다:", e)

def main():

    input = 'v_shooting_01_01.mpg'
    output = 'v_shooting_01_01.mp4'
    convert_mpg_to_mp4(input, output)
    """
    #base_folder = '/scratch2/dreamyou070/MyData/video/UCF11/UCF11_updated_mpg/UCF11_updated_mpg'
    base_folder = 'data/UCF11_updated_mpg'
    actions = os.listdir(base_folder)

    #save_folder = '/scratch2/dreamyou070/MyData/video/UCF11/UCF11_updated_mp4/UCF11_updated_mp4'
    save_folder = 'data/UCF11_updated_mp4'
    os.makedirs(save_folder, exist_ok=True)

    for action in actions:
        print(f"Converting {action}...")
        action_folder = os.path.join(base_folder, action)

        action_save_folder = os.path.join(save_folder, action)
        os.makedirs(action_save_folder, exist_ok=True)

        sub_folders = os.listdir(action_folder)
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(action_folder, sub_folder)
            sub_folder_save_path = os.path.join(action_save_folder, sub_folder)
            os.makedirs(sub_folder_save_path, exist_ok=True)

            if 'Annotation' not in sub_folder:
                mpg_files = os.listdir(sub_folder_path)

                for mpg_file in mpg_files:

                    mpg_file_path = os.path.join(sub_folder_path, mpg_file)
                    mp4_file_path = os.path.join(sub_folder_save_path, mpg_file.replace('.mpg', '.mp4'))
                    convert_mpg_to_mp4(mpg_file_path, mp4_file_path)
    """
if __name__ == '__main__':
    main()
