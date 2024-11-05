import os
import matplotlib.pyplot as plt
def main() :
    save_folder = '/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3_save_fig'
    os.makedirs(save_folder, exist_ok=True)
    print(f' step 1. read files')
    
    base_folder = '/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3_save'
    files = os.listdir(base_folder)
    for file in files :
        
        layer_timestep = file.split('_sorted_index')[0]
        layer_name = layer_timestep.split('_time')[0]
        time_info = layer_timestep.split(f'{layer_name}_')[-1]
        
        path = os.path.join(base_folder, file)
        with open(path, 'r') as f :
            content = f.readlines()
        content = content[1:]
        # make bar graph
        plt.figure(figsize=(20,10))
        x_axis, y_axis = [], []
        for line in content :
            line = line.strip()
            index, importance = line.split(' : ')
            x_axis.append(str(index))
            y_axis.append(float(importance))
        plt.bar(x_axis, y_axis)
        # angle
        plt.xlabel('dimension index')
        plt.ylabel('importance score')
        plt.title(f'{layer_name} ({time_info})')
        # x axis rotate
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_folder, f'{layer_name}_{time_info}.png'))
        # clear
        plt.clf
        plt.close()
    
if __name__ == "__main__" :
    main()
