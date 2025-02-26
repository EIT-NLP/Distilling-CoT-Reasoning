# Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning
![SVAMP](https://img.shields.io/badge/Dataset-SVAMP-blue)
![GSM8K](https://img.shields.io/badge/Dataset-GSM8K-blue)
![AQuA--RAT](https://img.shields.io/badge/Dataset-AQuA--RAT-blue)
![MATH](https://img.shields.io/badge/Dataset-MATH-blue)
![CommonsenseQA](https://img.shields.io/badge/Dataset-CommonsenseQA-blue)
![OpenbookQA](https://img.shields.io/badge/Dataset-OpenbookQA-blue)
![StrategyQA](https://img.shields.io/badge/Dataset-StrategyQA-blue)

![LLaMA-3.2-1B](https://img.shields.io/badge/Model-LLaMA--3.2--1B-21C2A4)
![LLaMA-3.2-3B](https://img.shields.io/badge/Model-LLaMA--3.2--3B-21C2A4)
![Gemma-2B](https://img.shields.io/badge/Model-Gemma--2B-21C2A4)
![BLOOM-560M](https://img.shields.io/badge/Model-BLOOM--560M-21C2A4)
![BLOOM-1.1B](https://img.shields.io/badge/Model-BLOOM--1.1B-21C2A4)
![BLOOM-1.7B](https://img.shields.io/badge/Model-BLOOM--1.7B-21C2A4)
![BLOOM-3B](https://img.shields.io/badge/Model-BLOOM--3B-21C2A4)

📰 [Paper](https://arxiv.org/pdf/2502.18001)

</div>

## 1. Introduction
Large Language Models (LLMs) excel in reasoning tasks through Chain-of-Thought (CoT) prompting. However, CoT prompting greatly increases computational demands, which has prompted growing interest in distilling CoT capabilities into Small Language Models (SLMs). This study systematically examines the factors influencing CoT distillation,  including the choice of granularity, format and teacher model. 

<p align="center">
  <img src="image/IntroFig.png" width="60%" />
  <p align="center">Overview of CoT Distillation. Different teacher models generate CoT supervision with varying levels of granularity and formats to fine-tune the student model.</p>
</p>


Through experiments involving four teacher models and seven student models across seven mathematical and commonsense reasoning datasets, we uncover three key findings: (1) Unlike LLMs, SLMs exhibit a *non-monotonic* relationship with granularity, with stronger models benefiting from finer-grained reasoning and weaker models performing better with simpler CoT supervision; (2) CoT format significantly impacts LLMs but has *minimal* effect on SLMs, likely due to their reliance on supervised fine-tuning rather than pretraining preferences; (3) Stronger teacher models do *NOT* always produce better student models, as diversity and complexity in CoT supervision can outweigh accuracy alone. These findings emphasize the need to tailor CoT strategies to specific student model, offering actionable insights for optimizing CoT distillation in SLMs.
