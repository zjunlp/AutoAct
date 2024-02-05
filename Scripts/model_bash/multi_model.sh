CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PEFT_SHARE_BASE_WEIGHTS=true python3 -m fastchat.serve.multi_model_worker \
    --port 31022 --worker http://localhost:31022 \
    --host localhost \
    --model-path /data/rolnan/FastChat/lora/hotpotqa/13b_test-hotpotqa-plan-5-epoch \
    --model-names "plan" \
    --model-path /data/rolnan/FastChat/lora/hotpotqa/13b_test-hotpotqa-action-5-epoch \
    --model-names "action" \
    --model-path /data/rolnan/FastChat/lora/hotpotqa/13b_test-hotpotqa-reflect-5-epoch \
    --model-names "reflect" \
    --max-gpu-memory 31Gib \
    --dtype float16 \
    --num-gpus 8