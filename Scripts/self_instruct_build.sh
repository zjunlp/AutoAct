python ../Self_Instruct/data_generation.py\
    --source_data  ../Self_Instruct/Meta_sample/Meta_Hotpotqa.json  \
    --target_data  ../Self_Instruct/hotpotqa_metaqa.json \
    --dataset_name hotpotqa  \
    --generate_all_num  800 \
    --generate_per_round_num  10 \
    --top_k 3\
    --model_name llama-2-13b-chat \
# python /data/rolnan/AutoAct/Self_Instruct/data_generation.py\
#     --source_data /data/rolnan/AutoAct/Self_Instruct/Meta_sample/Meta_Scienceqa.json \
#     --target_data  /data/rolnan/AutoAct/Self_Instruct/science_metaqa.json \
#     --dataset_name scienceqa  \
#     --generate_all_num  20 \
#     --generate_per_round_num  5 \
#     --top_k 3\
#     --model_name llama-2-7b-chat \