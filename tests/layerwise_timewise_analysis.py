import os

def main() :
    
    print(f' step 1. read files')
    base_folder = '/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3'
    save_folder = '/scratch2/dreamyou070/Prun/result/20241013_down00_pruning_ratio_erasing_layer_output_not_weight/pruned_model/pruning_ratio_0.3_save'
    os.makedirs(save_folder, exist_ok=True)
    files = os.listdir(base_folder)
    for file in files :
        name, ext = os.path.splitext(file)
        index_importance_dict = {}
        if ext == '.txt' :
            path = os.path.join(base_folder, file)
            with open(path, 'r') as f :
                content = f.readlines()
            for line in content :
                line_list = line.replace('[','').replace(']','')
                line_list = line_list.strip()
                line_list = line_list.split(', ')
                for idx, value in enumerate(line_list) :
                    if value not in index_importance_dict.keys() :
                        index_importance_dict[value] = idx
                    else :
                        index_importance_dict[value] += idx
            # save file
            save_path = os.path.join(save_folder, f'{name}_importance.txt')
            # sort index_importance_dict by value
            # from small to large
            sorted_index_importance_dict = dict(sorted(index_importance_dict.items(), key=lambda x : x[1]))
            # key = index_value
            # idx = importance
            with open(save_path, 'w') as f :
                f.write(f'dimension_index : importance_score \n')
                for index_value, importance in sorted_index_importance_dict.items() :                
                    f.write(f'{index_value} : {str(int(importance)/50)} \n')
            
                    
                
        
    
if __name__ == '__main__':
    main()