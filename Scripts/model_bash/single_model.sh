CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m fastchat.serve.model_worker \
    --port 31021 --worker http://localhost:31021 \
    --host localhost \
    --model-names "llama-2-13b-chat,text-embedding-ada-002" \
    --model-path /data/qiaoshuofei/PLMs/13b-chat \
    --max-gpu-memory 31Gib \
    --dtype float16 \
    --num-gpus 8