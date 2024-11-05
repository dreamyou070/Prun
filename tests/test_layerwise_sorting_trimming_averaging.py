import os

def main() :
    
    print(f' step 1. ')
    base_folder = '/scratch2/dreamyou070/Prun/result/20241017_layerwise_sorting_2'
    files = os.listdir(base_folder)
    layer_dict = {}
    for file in files :
        if 'down_blocks_3_motion_modules_0_proj_in' in file :
        
            layer_name = file.split('_time')[0]
            if layer_name not in layer_dict :
                layer_dict[layer_name] = {}
            file_path = os.path.join(base_folder, file)
            with open(file_path, 'r') as f :
                lines = f.readlines()
            for line in lines :
                line = line.strip()
                index, importance = line.split(' : ')
                if index not in layer_dict[layer_name] :
                    layer_dict[layer_name][index] = float(importance)
                else :  
                    layer_dict[layer_name][index] += float(importance)
                    
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        # save folder
        save_folder = '/scratch2/dreamyou070/Prun/result/20241017_layerwise_sorting_3'
        os.makedirs(save_folder, exist_ok=True)
        for layer_name in layer_dict :
            sorted_layer_dict = sorted(layer_dict[layer_name].items(), key=lambda x: x[1])
            with open(os.path.join(save_folder, f'{layer_name}_importance.txt'), 'w') as f :
                for key, value in sorted_layer_dict :
                    
                    #value = float(value.strip()) / 6
                    value = float(value) / 6
                    f.write(f'{key} : {value}\n')
    
if __name__ == "__main__" :
    main()