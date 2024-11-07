import os
import pandas as pd
def main() :

    #sample_file = '/scratch2/dreamyou070/MyData/video/openvid_1M/sample.csv'
    #sample_df = pd.read_csv(sample_file)
    #print(sample_df.head())

    origin_csv_file = '/scratch2/dreamyou070/MyData/video/openvid_1M/OpenVid-1M.csv'
    # video,caption,aesthetic score,motion score,temporal consistency score,camera motion,frame,fps,seconds
    df = pd.read_csv(origin_csv_file)
    ref_videoid = df['video'].tolist()
    ref_caption = df['caption'].tolist()



    sample_folder = '/scratch2/dreamyou070/MyData/video/openvid_1M/sample'
    subfolders = os.listdir(sample_folder)
    total = []
    for subfolder in subfolders:
        subfolder_path = os.path.join(sample_folder, subfolder)
        files = os.listdir(subfolder_path)
        for file in files:
            videoid = file
            page_dir = f'{subfolder}/{file}'
            trg_idx = ref_videoid.index(videoid)
            caption = ref_caption[trg_idx]
            elem = [videoid, page_dir, caption]
            total.append(elem)
    # make df
    header = ['videoid', 'page_dir', 'name']
    file = '/scratch2/dreamyou070/MyData/video/openvid_1M_sample.csv'
    df = pd.DataFrame(total, columns=header)
    df.to_csv(file, index=False)


    """
    file = '/scratch2/dreamyou070/MyData/video/openvid_1M_sample.csv'
    df = pd.read_csv(file)
    page_dir = df['page_dir'].tolist(
    new_dirs = []
    for dir in page_dir:
        if 'video/openvid_1M/' in dir :
            # insert 'sample' after front
            front = dir.split('openvid_1M/')[0]
            back = dir.split('openvid_1M/')[1]
            new_dirs.append(back)
            print(new_dirs)
            break
        else :
            new_dirs.append(dir)
    """

if __name__ == '__main__':
    main()