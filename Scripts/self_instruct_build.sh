python Self_Instruct/data_generation.py \
    --source_data Self_Instruct/Meta_sample/Meta_Hotpotqa.json \
    --target_data Self_Instruct/hotpotqa_metaqa.json \
    --dataset_name hotpotqa  \
    --generate_all_num 800 \
    --generate_per_round_num 10 \
    --model_name llama-2-13b-chat \