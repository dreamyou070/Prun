import os

def main():
    save_folder = '/scratch2/dreamyou070/Prun/result/20241017_layerwise_sorting_2'
    os.makedirs(save_folder, exist_ok=True)
    base_folder = '/scratch2/dreamyou070/Prun/result/20241017_layerwise_sorting_1'
    files = os.listdir(base_folder)
    for file in files :
        print(file)
        file_dict = {}
        file_path = os.path.join(base_folder, file)
        with open(file_path, 'r') as f :
            contents = f.readlines()
        total_data_num = len(contents)
        g = 0
        for line in contents :
            line = line.strip()
            
            if len(line) > 0 :
                
                g += 1
                line = line.replace('[','')
                line = line.replace(']','')
                line_list = line.split(',') # list of string
                for index, dimension in enumerate(line_list) :
                    dimension = dimension.strip()
                    #if file == 'up_blocks_1_motion_modules_0_transformer_blocks_0_attn1_to_q_time_3.txt' :
                    #    print(dimension)
                    dimension = int(dimension)
                    
                    if dimension not in file_dict :
                        file_dict[dimension] = index
                    else :
                        file_dict[dimension] += index
        # averaging
        for key in file_dict :
            file_dict[key] = file_dict[key] / g
        # sorting by value (from small to large)
        sorted_file_dict = sorted(file_dict.items(), key=lambda x: x[1])
        with open(os.path.join(save_folder, file), 'w') as f :
            for key, value in sorted_file_dict :
                f.write(f'{key} : {value}\n')
            
if __name__ == '__main__':
    main()