import os
from datetime import datetime

dataset_name = "GSM8k_llama_level1"  

gpu_devices = 4  
models = [
    # "/code/huggingface/hub/Gemma-2b",
    # "/code/huggingface/hub/bloom-560m",
    # "/code/huggingface/hub/bloom-1b1",
    # "/code/huggingface/hub/bloom-1b7",
    # "/code/huggingface/hub/bloom-3b",
    # "/code/huggingface/hub/Llama-3.2-1B",
    "/code/huggingface/hub/Llama-3.2-3B"
]

def generate_yaml(dataset_name, gpu_devices, models):
    today_date = datetime.now().strftime("%m%d")
    
    config_folder = os.path.join(dataset_name)
    os.makedirs(config_folder, exist_ok=True)

    model_templates = {
        "gemma": "gemma",
        "bloom": "alpaca",
        "llama3": "llama3",
    }
    
    total_batch_size = 64

    for model_path in models:
        model_name = model_path.split("/")[-1]
        
        if "gemma" in model_name.lower():
            model_key = "gemma"
        elif "bloom" in model_path.lower(): 
            model_key = "bloom"
        elif "llama" in model_name.lower():
            model_key = "llama3"
        else:
            raise ValueError(f"Can not identify: {model_path}")
        
        template = model_templates[model_key]
        
        per_device_train_batch_size = 4  # fixed value
        gradient_accumulation_steps = total_batch_size // (gpu_devices * per_device_train_batch_size)
        
        output_dir = f"{today_date}/{model_name}/{dataset_name}"
        
        yaml_content = f"""\
### model
model_name_or_path: {model_path}

stage: sft
do_train: true
finetuning_type: full
flash_attn: auto

### ddp
ddp_timeout: 180000000
deepspeed: cache/ds_z3_config.json

### dataset
dataset: {dataset_name}

template: {template}

cutoff_len: 4096
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16
seed: 100
### output
output_dir: {output_dir}

logging_steps: 5
# save_strategy: epoch
plot_loss: true
overwrite_output_dir: true
buffer_size: 1

### train 
per_device_train_batch_size: {per_device_train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: 5.0e-05
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_steps: 0
fp16: False
max_grad_norm: 1.0
optim: adamw_torch 
packing: False

### validation
# val_size: 0.1
# eval_steps: 20
# evaluation_strategy: steps
# load_best_model_at_end: true 

### wandb
report_to: all
"""
        yaml_filename = f"{model_key.capitalize()}_{model_name}_{dataset_name}.yaml"
        yaml_path = os.path.join(config_folder, yaml_filename)
        
        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(yaml_content)
        
        print(f"Generated: {yaml_path}")

generate_yaml(dataset_name, gpu_devices, models)
