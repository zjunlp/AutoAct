<h1 align="center"> AutoAct </h1>
<h3 align="center"> Automatic Agent Learning from Scratch via Self-Planning </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2401.05268">📄arXiv</a> •
  <a href="https://huggingface.co/papers/2401.05268">🤗HFPaper</a> •
  <a href="http://easydetect.openkg.cn/">🌐Web</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/AutoKG) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/AutoAct?color=green) 

## Table of Contents

- 🌻[Acknowledgement](#🌻acknowledgement)
- 🌟[Overview](#🌟overview)
- 🔧[Installation](#🔧installation)
- ✏️[Self-Instruct](#✏️Self-Instruct)
- 📝[Self-Planning](#📝Self-Planning)
  - [Automatic Tool Selection](#Automatic-Tool-Selection)
  - [Trajectories Synthesis](#Trajectories-Synthesis)
  - [Self-Differentiation](#Self-Differentiation)
  - [Group Planning](#Group-Planning)
- 🚩[Citation](#🚩Citation)

---



## 🌻Acknowledgement

Our code of training module is referenced and adapted from [FastChat](https://github.com/lm-sys/FastChat), while the code of inference module is implemented based on [BOLAA](https://github.com/salesforce/BOLAA). Various baseline codes use [ReAct](https://github.com/ysymyth/ReAct), [Reflexion](https://github.com/noahshinn/reflexion), [BOLAA](https://github.com/salesforce/BOLAA), [Chameleon](https://github.com/lupantech/chameleon-llm), [ReWOO](https://github.com/billxbf/ReWOO), [FireAct](https://github.com/anchen1011/FireAct) respectively. We use LangChain with open models via [Fastchat](https://github.com/lm-sys/FastChat/blob/main/docs/langchain_integration.md). Thanks for their great contributions!



## 🌟Overview

Language agents have achieved considerable performance on various complex tasks. Despite the incessant exploration in this field, existing language agent systems still struggle with costly, non-reproducible data reliance and face the challenge of compelling a single model for multiple functions. To this end, we introduce **AutoAct**, an automatic agent learning framework that does not rely on large-scale annotated data and synthetic trajectories from closed-source models (e.g., GPT-4). Given limited data with a tool library, **AutoAct** first automatically synthesizes planning trajectories without any assistance from humans or strong closed-source models. Then, **AutoAct** leverages a *division-of-labor* strategy to automatically differentiate based on the target task information and synthesized trajectories, producing a sub-agent group to complete the task. We conduct comprehensive experiments with different LLMs, which demonstrates that **AutoAct** yields better or parallel performance compared to various strong baselines.

<img src="/Users/qsf/work/academic/paper/ACL24/writing/method.gif" alt="method" style="zoom: 50%;" />



## 🔧Installation

```bash
git clone https://github.com/zjunlp/AutoAct
cd AutoAct
pip install -r requirements.txt
```



## ✏️Self-Instruct

We conduct self-instruct on Meta-Agent to acquire a sufficient amount of task data and provide an ample training resource. 

```bash
python Self_Instruct/data_generation.py \
    --source_data Self_Instruct/Meta_sample/Meta_Hotpotqa.json \
    --target_data Self_Instruct/hotpotqa_metaqa.json \
    --dataset_name hotpotqa  \
    --generate_all_num 800 \
    --generate_per_round_num 10 \
    --model_name llama-2-13b-chat \
```

The `source_data` contains data examples from the target task information. The `target_data` consists of data generated through self-instruct. The variable `generate_all_num` represents the total number of generated data instances. In order to improve generation efficiency and avoid duplication, we generate `generate_per_round_num` data instances per round.



## 📝Self-Planning

### Automatic Tool Selection

With the tool library at hand, we ask the Meta-Agent to select applicable tools for each task automatically.

```bash
python Self_Planning/Tool_Selection/tool_selected.py \
    --model_name llama-2-13b-chat \
    --task_name ScienceQA \
    --top_k 40 \
    --top_p 0.75 \
    --max_tokens 1024 \
    --tool_save_path Self_Planning/Tool_Selection/{task_name}_Tools.json
```

The information of the selected tools will be stored in `tool_save_path`.



### Trajectories Synthesis

```bash
python Self_Planning/Traj_Syn/run_task.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --llm_name llama-2-13b-chat \
    --max_context_len 4096 \
    --task Hotpotqa \
    --task_path Self_Instruct/hotpotqa_metaqa.json \
    --save_path Self_Planning/Traj_Syn/output/hotpotqa_train_data.jsonl
```

In order to obtain high-quality synthesized trajectories, we filter out all the trajectories with $\texttt{reward}<1$ and collect trajectories with exactly correct answers ($\texttt{reward}=1$) as the training source for self-differentiation.

```bash
python Scripts/filter_data.py \
    --source_path Self_Planning/Traj_Syn/output/hotpotqa_train_data.jsonl \
    --save_path Self_Planning/Traj_Syn/output \
    --task_name HotpotQA \
    --filter_num 200
```



### Self-Differentiation

In order to establish a clear *division-of-labor*, we leverage synthesized planning trajectories to differentiate the Meta-Agent into three sub-agents with distinct functionalities:

- **Plan-Agent** undertakes task decomposition and determines which tool to invoke in each planning loop.
- **Tool-Agent** is responsible for how to invoke the tool by deciding the parameters for the tool invocation.
- **Reflect-Agent** engages in reflection by considering all the historical trajectories and providing a reflection result.

Agent training:

```bash
for agent in plan tool reflect
do
echo "####################"
echo $agent
echo "####################"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /Self-Planning/Train/train_lora.py \
    --model_name_or_path llama-2-13b-chat \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path Self_Planning/Traj_Syn/output/data_$agent.json \
    --output_dir /Self-Planning/Train/lora/HotpotQA/13b-$agent-5-epoch \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fp16 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --q_lora False \
    --deepspeed playground/deepspeed_config_s3.json \
    --resume_from_checkpoint False 
done
```



### Group Planning

After obtaining the task-specific sub-agents, any new question is processed through group planning among the sub-agents to achieve the desired outcome.

```bash
python Self_Planning/Group_Planning/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent plan \
    --tool_agent tool \
    --reflect_agent reflect \
    --max_context_len 4096 \
    --task HotpotQA \
    --task_path Self_Planning/Group_Planning/benchmark_run/data/hotpotqa \
    --save_path Self_Planning/Group_Planning/output/13b
```



## 🚩Citation

Please cite our repository if you use AutoAct in your work. Thanks!

```bibtex
@article{DBLP:journals/corr/abs-2401-05268,
  author       = {Shuofei Qiao and
                  Ningyu Zhang and
                  Runnan Fang and
                  Yujie Luo and
                  Wangchunshu Zhou and
                  Yuchen Eleanor Jiang and
                  Chengfei Lv and
                  Huajun Chen},
  title        = {{AUTOACT:} Automatic Agent Learning from Scratch via Self-Planning},
  journal      = {CoRR},
  volume       = {abs/2401.05268},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2401.05268},
  doi          = {10.48550/ARXIV.2401.05268},
  eprinttype    = {arXiv},
  eprint       = {2401.05268},
  timestamp    = {Thu, 25 Jan 2024 15:41:08 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2401-05268.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



## 🎉Contributors

<a href="https://github.com/zjunlp/AutoAct/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=zjunlp/AutoAct" /></a>

We will offer long-term maintenance to fix bugs and solve issues. So if you have any problems, please put issues to us.
