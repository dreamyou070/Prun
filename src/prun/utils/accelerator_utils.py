from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from accelerate import DistributedDataParallelKwargs
from datetime import datetime, timedelta
import os

def make_accelerator(args) :
    
    if args.add_qkv :
        args.find_unused_parameters = True # there is unused parameters 
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=30000))
    # gradient_accumulation_steps=args.gradient_accumulation_steps
    # 설명: Gradient accumulation steps를 설정합니다. 이 값은 매 업데이트마다 몇 번의 배치의 gradient를 모아서 업데이트할지를 결정합니다. 예를 들어, gradient_accumulation_steps가 4로 설정되어 있다면, 4개의 배치를 처리한 후에 gradient를 업데이트합니다.
    # mixed_precision=args.mixed_precision
    # 설명: Mixed precision 학습을 사용할지 여부를 설정합니다. Mixed precision 학습은 FP16과 FP32를 혼합하여 사용하여 메모리 사용량을 줄이고 성능을 향상시킬 수 있습니다.
    # log_with=args.report_to
    # 설명: 로그를 기록할 방법을 지정합니다. args.report_to 값에 따라 로그가 기록되는 방법(예: TensorBoard, WandB 등)이 결정됩니다.
    # project_config=accelerator_project_config
    # 설명: Accelerator 프로젝트 설정을 포함하는 객체를 전달합니다. 이 설정은 프로젝트와 관련된 다양한 구성을 정의합니다.
    # split_batches=True
    # 설명: 배치를 분할할지 여부를 결정합니다. WebDataset을 사용하는 경우, 이 값을 True로 설정하는 것이 중요합니다. split_batches=True일 때는 각 프로세스가 배치를 나누어 사용하고, 학습률 스케줄링에 올바른 스텝 수를 반영할 수 있습니다. 이 값을 False로 설정하면, 배치가 프로세스 수에 따라 나뉘며, 배치 크기가 프로세스 수에 의해 곱해져서 스텝 수가 나뉘어질 수 있습니다.
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                                mixed_precision=args.mixed_precision,
                                log_with=args.report_to,
                                kwargs_handlers=[kwargs,ddp_kwargs],)
    return accelerator

def get_folder(output_dir, sub_folder_name): 
    # [1]     
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, sub_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # [2] sample folder
    custom_save_folder = os.path.join(output_dir, "custom_samples")
    os.makedirs(custom_save_folder, exist_ok=True)
    eval_save_folder = os.path.join(output_dir, "eval_samples")
    os.makedirs(eval_save_folder, exist_ok=True)

    # [3] 
    sanity_folder = f"{output_dir}/sanity_check"
    os.makedirs(sanity_folder, exist_ok=True)

    # [4] 
    model_save_dir = f"{output_dir}/checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)

    # [5] log_dir
    log_folder = os.path.join(output_dir, "logs")
    os.makedirs(log_folder, exist_ok=True)
    
    print(f' in get folder, output_dir = {output_dir}')

    return output_dir, custom_save_folder, eval_save_folder, sanity_folder, model_save_dir, log_folder

def make_skip_layers_dot(skip_layers) :
    # make skip_layers_dot : "up_blocks_0_motion_modules_0"
    # "mid_block_motion_modules_0",
    skip_layers_dot = []
    for layer in skip_layers:
        layer_dot_list = layer.split('_') # mid/blocks_motion_modules.0
        dot_layer = ''
        for e in layer_dot_list :
            try :
                int_e = str(int(e))
                int_e = f'.{int_e}.'
            except :
                int_e = f'_{e}_' # _mid_blocks.0.motion_modules.0.
            dot_layer += int_e
        dot_layer = dot_layer.replace('__','_')
        dot_layer = dot_layer.replace('_.','.')
        dot_layer = dot_layer.replace('._','.')
        if dot_layer.startswith('_'):
            dot_layer = dot_layer[1:]
        if dot_layer.endswith('_'):
            dot_layer = dot_layer[:-1]
        if dot_layer.endswith('.'):
            dot_layer = dot_layer[:-1]
        if 'mid' in dot_layer :
            dot_layer = dot_layer.replace('_motion', '.motion')
        skip_layers_dot.append(dot_layer)
    return skip_layers_dot