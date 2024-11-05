# csv file making
import json
import os


base = '/home/dreamyou070/VideoDistill/experiment_8_training_free'
main_folder = 'sharp_attnmap_layers_12'
base_dir = os.path.join(base, main_folder)
file_list = os.listdir(base_dir)
for file in file_list:
    # if json file
    name, ext = os.path.splitext(file)
    if ext == '.json':
        file_dir = os.path.join(base_dir, file)
        if 'eval' in name:
            new_folder = os.path.join(base_dir, f'evaluation_result_folder')
        else:
            new_folder = os.path.join(base_dir, f'evaluation_info_folder')
        os.makedirs(new_folder, exist_ok=True)
        new_file = os.path.join(new_folder, file)
        # if eval in name
        os.rename(file_dir, new_file)
# get only eval_result json file
result_folder = os.path.join(base_dir, 'evaluation_result_folder')
file_list = os.listdir(result_folder)
result_dict = {}
for file in file_list:
    name, ext = os.path.splitext(file)
    name_list = name.split('_')
    epoch_info = name_list[0]
    # open file
    file_dir = os.path.join(result_folder, file)
    with open(file_dir, 'r') as f:
        json_data = json.load(f)
    dimension = list(json_data.keys())[0]
    value = json_data[dimension][0]
    if epoch_info not in result_dict.keys():
        result_dict[epoch_info] = {}
    result_dict[epoch_info][dimension] = value
# save csv file
import pandas as pd
# when make csv file, epoch_info ordering and dimension ordering
epoch_list = list(result_dict.keys())
epoch_list.sort()
dimension_list = list(result_dict[epoch_list[0]].keys())
dimension_list.sort()
result_dict_ordered = {}
for epoch in epoch_list:
    result_dict_ordered[epoch] = {}
    for dimension in dimension_list:
        result_dict_ordered[epoch][dimension] = result_dict[epoch][dimension]
result_dict = result_dict_ordered
# whant to index and column transpose
df = pd.DataFrame(result_dict)
df = df.T
df.to_csv(os.path.join(base_dir, f'{main_folder}_result.csv'))


