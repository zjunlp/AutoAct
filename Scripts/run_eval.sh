python Self_Planning/Group_Planning/run_eval.py \
    --agent_name ZeroshotThink_HotPotQA_run_Agent \
    --plan_agent plan \
    --tool_agent tool \
    --reflect_agent reflect \
    --max_context_len 4096 \
    --task HotpotQA \
    --task_path Self_Planning/Group_Planning/benchmark_run/data/hotpotqa \
    --save_path Self_Planning/Group_Planning/output/13b