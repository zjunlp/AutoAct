python Self_Plan/Traj_Syn/run_task.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --llm_name llama-2-13b-chat \
    --max_context_len 4096 \
    --task Hotpotqa \
    --task_path Self_Instruct/hotpotqa_metaqa.json \
    --save_path Self_Plan/Traj_Syn/output/hotpotqa_train_data.jsonl