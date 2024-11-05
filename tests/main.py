import os

csv_path = '/scratch2/dreamyou070/MyData/video/openvid_1M/sample.csv'
with open(csv_path, 'r') as f:
    lines = f.readlines()

for line in lines :
    print(line)
    break

sample_folder =  '/scratch2/dreamyou070/MyData/video/openvid_1M/sample'
folders = os.listdir(sample_folder)
for folder in folders :
    folder_dir = os.path.join(sample_folder, folder)
    files=os.listdir(folder_dir)
    print(f'len of files = {len(files)}')
