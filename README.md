<h1 align="center"><b>Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning</b></h1>
</div>

<p align="center">
<a href="https://arxiv.org/pdf/2502.18001">
  <img src="https://img.shields.io/badge/Arxiv-2502.18001-orange.svg"></a> 
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a> 
<a href="https://github.com/EIT-NLP/Distilling-CoT-Reasoning/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>

## Introduction

Large Language Models (LLMs) excel in reasoning tasks through Chain-of-Thought (CoT) prompting. However, CoT prompting greatly increases computational demands, which has prompted growing interest in distilling CoT capabilities into Small Language Models (SLMs). This study systematically examines the factors influencing CoT distillation,  including the choice of **granularity**, **format** and **teacher model**. 

<p align="center">
  <img src="image/Intro.jpg" width="60%"/>
  <p align="center">Overview of CoT Distillation. Different teacher models generate CoT supervision with varying levels of granularity and formats to fine-tune the student model.</p>
</p>


Through experiments involving four teacher models and seven student models across seven mathematical and commonsense reasoning datasets, we uncover three key findings: (1) Unlike LLMs, SLMs exhibit a ***non-monotonic*** relationship with granularity, with stronger models benefiting from finer-grained reasoning and weaker models performing better with simpler CoT supervision; (2) CoT format significantly impacts LLMs but has ***minimal*** effect on SLMs, likely due to their reliance on supervised fine-tuning rather than pretraining preferences; (3) Stronger teacher models do ***NOT*** always produce better student models, as diversity and complexity in CoT supervision can outweigh accuracy alone. These findings emphasize the need to tailor CoT strategies to specific student model, offering actionable insights for optimizing CoT distillation in SLMs.

## Todo

- [x] Release evaluation code on math and commonsense reasoning
- [x] Release SFT datasets
- [x] Add instructions for SFT on LLaMA-Factory

## Experiments Setup

We conducted extensive experiments on **four mathematical** reasoning datasets of varying difficulty and **three commonsense** reasoning datasets, using **four teacher models** to distill reasoning skills to **seven student models**. 

### Datasets

| **Training Dataset**                                         | **Samples (Training)** | **Samples (Testing)** | **Fields**                                                  | **Human Annotation** |
| ------------------------------------------------------------ | ---------------------- | --------------------- | ----------------------------------------------------------- | -------------------- |
| [SVAMP](https://huggingface.co/datasets/ChilleD/SVAMP)       | 700                    | 300                   | Arithmetic problems                                         | Yes                  |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k)        | 7.4k                   | 1.3k                  | Grade-school math                                           | Yes                  |
| [AQuA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat) | 6.1k                   | 254                   | Algebraic reasoning, multi-step                             | Yes                  |
| [Math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | 1.3k                   | 500                   | Pre-Algebra, Algebra, Counting & Probability, Number Theory | Yes                  |
| [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | 9.7k                   | 1.2k                  | Commonsense knowledge                                       | Yes                  |
| [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) | 4.9k                   | 500                   | Domain-specific knowledge                                   | No                   |
| [StrategyQA](https://github.com/eladsegal/strategyqa)        | 2k                     | 290                   | Multi-step reasoning                                        | Yes                  |

### Models

Teacher models: [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Gemini-1.5-Flash](https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/), [LLaMA 3 70B](https://ai.meta.com/blog/meta-llama-3/)

Student models: [LLaMA 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [LLaMA 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B), [Gemma 2B](https://huggingface.co/google/gemma-2b), BLOOM [560M](https://huggingface.co/bigscience/bloom-560m), [1.1B](https://huggingface.co/bigscience/bloom-1b1), [1.7B](https://huggingface.co/bigscience/bloom-1b7), [3B](https://huggingface.co/bigscience/bloom-3b)

## Installation

Our experiment uses a pipeline of **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main)** to fine-tune the student models.

> [!IMPORTANT]
> Installation is mandatory.

```bash
conda create -n llama_factory python==3.10
conda activate llama_factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

For the evaluation environment:

```bash
conda create -n evaluation python==3.10
conda activate evaluation
cd Evaluation
pip install -r requirements.txt
```

## Training

The training data are provided in the `data` folder. Please refer to [data/readme.md](https://github.com/EIT-NLP/Distilling-CoT-Reasoning/blob/main/data/readme.md) to see a list of our datasets. Here's how to set up training:

1. After cloning the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main) repository, copy **all contents** from this repository's `data` folder into the `data` folder of the LLaMA Factory directory.

2. We provide training configs generation code `config/yamlgeneration.py`. You can modify `dataset_name`, `gpu_devices`, and `models` and then run the following command to generate configs:

```bash
cd config
python yamlgeneration.py
```

3. To fine-tune the target LLM, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config/<your_dataset_name>/<models>_<your_dataset_name>.yaml
```

Or you can run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

for config_file in /code/LLaMA-Factory/config/<your_dataset_name>/*.yaml; do
    llamafactory-cli train "$config_file"
done
```

to train on multiple configurations continuously.

> We provide our training configs in `config/examples` for your reference.

## Evaluation

The evaluation code is built from [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH).

### Single Evaluation Run

To perform a single evaluation, use the following commands:

For Mathematical Reasoning:

```bash
CUDA_VISIBLE_DEVICES=0 python run_open.py \
    --model path_to_your_model \
    --shots 0 \
    --dataset your_dataset_name \
    --model_max_length 1024 \
    --dtype bfloat16 \
    --form your_model_form
```

For Commonsense Reasoning:

```bash
CUDA_VISIBLE_DEVICES=0 python run_reasoning.py \
    --model path_to_your_model \
    --dataset your_dataset_name \
    --output test.json \
    --model_max_length 640 \
    --dtype bfloat16 \
    --form your_model_form
```

Arguments Explanation:

`--model`: Path to your fine-tuned model.
`--shots`: Number of few-shot examples (set to 0 for zero-shot evaluation).
`--dataset`: Name of the dataset (see valid options below).
`--model_max_length`: Maximum sequence length.
`--dtype`: Data type for evaluation.
`--form`: Model template (choose from gemma, llama, alpaca).

`dataset` Options: 

Mathematical Reasoning Datasets: svamp, gsm8k, aqua, math
Commonsense Reasoning Datasets: csqa_test.json, openbookQA_test.json, strategyQA_test.json

### Batch Evaluation

To run large-scale evaluations across multiple models:

Modify the following parameters in evaluate_models.py or autoevaluate.py:

- num_gpus: Number of GPUs to utilize.
- output_file: Path to save the evaluation results.
- model_dir: Directory containing the models to evaluate.

Run the respective evaluation scripts:

```bash
# For Mathematical Reasoning:
python evaluate_models.py
# For Commonsense Reasoning:
python autoevaluate.py
```

## Acknowledgments

The evaluation code is built from [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH).

## Citation

```bibtex
@misc{chen2025unveilingkeyfactorsdistilling,
      title={Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning}, 
      author={Xinghao Chen and Zhijing Sun and Wenjin Guo and Miaoran Zhang and Yanjun Chen and Yirong Sun and Hui Su and Yijie Pan and Dietrich Klakow and Wenjie Li and Xiaoyu Shen},
      year={2025},
      eprint={2502.18001},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18001}, 
}
```

## Contact

If you have any questions, feel free to raise an issue or contact us at <xing-hao.chen@connect.polyu.hk>.