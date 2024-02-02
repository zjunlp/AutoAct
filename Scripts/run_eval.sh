python ../Self_Plan/Tarject_Plan/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent llama-2-13b-chat \
    --action_agent llama-2-13b-chat \
    --reflect_agent llama-2-13b-chat \
    --max_context_len 1024 \
    --task Hotpotqa \
    --task_path /data/rolnan/AutoAct/Self_Plan/Tarject_Plan/benchmark_run/data/hotpotqa\
    --save_path ../Self_Plan/Tarject_Plan/output/hotpotqa_data.json\