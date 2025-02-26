import os
import argparse
import threading
import subprocess
import re
from glob import glob

num_gpus = 4

def parse_args():
    parser = argparse.ArgumentParser(description='自动化模型测评脚本。')
    parser.add_argument('--model_dir', default='/code/LLaMA-Factory/1210reason', help='模型的基目录。')
    parser.add_argument('--output_file', default='1210human.txt', help='测评结果汇总文件。')
    return parser.parse_args()

def find_model_directories(base_dir):
    # 查找所有两级深度的目录
    pattern = os.path.join(base_dir, '*/*')
    model_dirs = [d for d in glob(pattern) if os.path.isdir(d)]
    return model_dirs

def determine_form(model_path):
    # 根据第一层子目录名称确定form
    base_name = os.path.basename(os.path.dirname(model_path))
    # print(base_name)
    if base_name.startswith('bloom'):
        return 'alpaca'
    elif base_name.startswith('Gemma'):
        return 'gemma'
    elif base_name.startswith('Llama'):
        return 'llama3'
    else:
        return 'llama3'  # 默认设置

def determine_dataset(model_dir):
    # 根据model_dir路径来确定数据集
    print(model_dir)
    if '/ecqa' in model_dir:
        return 'csqa_test.json'
    elif '/csqa' in model_dir:
        return 'csqa_test.json'
    elif '/cQA' in model_dir:
        return 'csqa_test.json'
    elif '/obqa' in model_dir:
        return 'openbookQA_test.json'
    elif '/stqa' in model_dir:
        return 'strategyQA_test.json'
    else:
        print("警告: 未能匹配到数据集，将使用默认数据集。")
        return 'default_test.json'

def generate_output_path(dataset_name, model_path):
    # 生成输出文件路径，基于运行文件夹的 output/datasetname/两级深度的目录.json
    model_subdir = os.path.basename(os.path.dirname(model_path)) + '_' + os.path.basename(model_path)
    output_dir = os.path.join('output', dataset_name, model_subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'result.json')
    return output_path

def evaluate_model(model_info):
    model_path = model_info['model_path']
    tokenizer_path = model_info['model_path']  # 本地tokenizer路径与模型路径相同
    form = model_info['form']
    dataset = model_info['dataset']
    gpu_id = model_info['gpu_id']
    output_path = generate_output_path(dataset.split('.')[0], model_path)  # 生成特定的输出路径

    command = [
        'python', 'run_reasoning.py',
        '--model', model_path,
        '--dataset', dataset,
        '--output', output_path,
        '--model_max_length', '640',
        '--dtype', 'bfloat16',
        '--form', form
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"开始测评模型: {os.path.basename(model_path)}，使用GPU: {gpu_id}")
    print("执行命令: " + ' '.join(command))

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        stdout, stderr = process.communicate()
        output = stdout.decode('utf-8', errors='ignore') + stderr.decode('utf-8', errors='ignore')

        # 解析输出
        token_length_pattern = r'Output token lengths - Max: (\d+), Min: (\d+), Avg: ([\d\.]+)'
        accuracy_pattern = r'Final accuracy:\s+([\d\.]+)'

        token_lengths = re.search(token_length_pattern, output)
        accuracy = re.search(accuracy_pattern, output)

        if token_lengths:
            max_len = token_lengths.group(1)
            min_len = token_lengths.group(2)
            avg_len = token_lengths.group(3)
        else:
            max_len = min_len = avg_len = 'N/A'

        if accuracy:
            final_accuracy = accuracy.group(1)
        else:
            final_accuracy = 'N/A'

        # 构造模型名称为 一级子文件夹名称 + '/' + 模型名称
        first_level_folder = os.path.basename(os.path.dirname(model_path))
        model_name = first_level_folder + '/' + os.path.basename(model_path)

        result = {
            'model_name': model_name,
            'dataset': dataset,
            'form': form,
            'max_len': max_len,
            'min_len': min_len,
            'avg_len': avg_len,
            'final_accuracy': final_accuracy
        }

        # 打印输出以进行调试
        print(f"模型 {result['model_name']} 测评完成。GPU: {gpu_id}")
        print("测评输出:")
        print(output)
        print("解析结果:", result)
        print("-" * 50)

    except Exception as e:
        print(f"模型 {model_path} 测评时出错: {e}")
        first_level_folder = os.path.basename(os.path.dirname(model_path))
        model_name = first_level_folder + '/' + os.path.basename(model_path)
        result = {
            'model_name': model_name,
            'dataset': dataset,
            'form': form,
            'max_len': 'Error',
            'min_len': 'Error',
            'avg_len': 'Error',
            'final_accuracy': 'Error'
        }

    return result

def worker(model_infos, output_file, lock):
    for model_info in model_infos:
        result = evaluate_model(model_info)
        # 在每个模型测评完成后，立即将结果写入汇总文件
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('模型名称: {}\n'.format(result['model_name']))
                f.write('数据集: {}\n'.format(result['dataset']))
                f.write('Form: {}\n'.format(result['form']))
                f.write('Max Token Length: {}\n'.format(result['max_len']))
                f.write('Min Token Length: {}\n'.format(result['min_len']))
                f.write('Avg Token Length: {}\n'.format(result['avg_len']))
                f.write('Final Accuracy: {}\n'.format(result['final_accuracy']))
                f.write('\n')

def main():
    args = parse_args()
    model_dirs = find_model_directories(args.model_dir)
    model_list = []
    for model_path in model_dirs:
        form = determine_form(model_path)
        dataset = determine_dataset(model_path)  # 自动确定数据集文件名
        model_list.append({
            'model_path': model_path,
            'form': form,
            'dataset': dataset,
            'tokenizer_path': model_path,  # 本地tokenizer路径与模型路径相同
            'output_file': args.output_file
        })

    gpu_model_lists = [[] for _ in range(num_gpus)]
    for idx, model_info in enumerate(model_list):
        gpu_id = idx % num_gpus
        model_info['gpu_id'] = gpu_id
        gpu_model_lists[gpu_id].append(model_info)

    threads = []
    lock = threading.Lock()

    # 在开始前清空输出文件内容
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write('')  # 清空文件

    for gpu_id in range(num_gpus):
        t = threading.Thread(target=worker, args=(gpu_model_lists[gpu_id], args.output_file, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
