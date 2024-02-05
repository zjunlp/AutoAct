python ../Self_Plan/Tarject_Plan/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent plan \
    --action_agent action \
    --reflect_agent reflect \
    --max_context_len 4096 \
    --task Hotpotqa \
    --task_path /data/rolnan/AutoAct/Self_Plan/Tarject_Plan/benchmark_run/data/hotpotqa \
    --save_path ../Self_Plan/Tarject_Plan/output/13b/tem_0.5\