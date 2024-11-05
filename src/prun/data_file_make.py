import os
import pandas as pd

# video_id, page_dir, name
def main():

    # save
    save_file = '/scratch2/dreamyou070/MyData/video/openvid_1M_sample.csv'
    header = ['videoid', 'page_dir', 'name']

    # [0]
    sample_file = f'/scratch2/dreamyou070/MyData/video/openvid_1M/OpenVid-1M.csv'
    df = pd.read_csv(sample_file)
    video_id = df['video'].tolist()
    name = df['caption'].tolist()


    # [1] basic folder
    basic_folder = '/scratch2/dreamyou070/MyData/video/openvid_1M'
    folders = os.listdir(basic_folder)
    total = []
    for folder_name in folders:
        if folder_name != 'download' and folder_name != 'sample.csv' and folder_name != 'sample' and folder_name != 'main.py' and '.csv' not in folder_name:
            folder_dir = os.path.join(basic_folder, folder_name)
            files = os.listdir(folder_dir) # here problem
            for file in files:

                try :
                    idx = video_id.index(file)
                    trg_name = name[idx].strip()
                    elem = [file, f'{folder_dir}/{file}', trg_name]
                    total.append(elem)
                except :
                    print(f'Error: {file}')
                    continue

    # make csv file with header
    df = pd.DataFrame(total, columns=header)
    df.to_csv(save_file, index=False)


if __name__ == '__main__':
    main()