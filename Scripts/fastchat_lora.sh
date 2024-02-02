for epoch in 5
do
for data in hotpotqa
do
for agent in plan 
do
for ft_num in 300
do
echo "####################"
echo $agent $epoch $data
echo "####################"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed /data/rolnan/FastChat/fastchat/train/train_lora.py \
    --model_name_or_path /PLMs/7b-chat \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /data/rolnan/scripts/train_data/70b/hotpotqa/ft_num/$ft_num/data_$agent.json \
    --output_dir ./lora/$data/7b-from_70_$ft_num-$data-$agent-$epoch-epoch \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 1 \
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
done
done
done