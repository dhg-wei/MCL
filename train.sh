randport=$(shuf -i8000-9999 -n1)  # Generate a random port number
CUDA_VISIBLE_DEVICES=1,2,3,4 python -u main.py \
    --dist-url "tcp://127.0.0.1:${randport}" --dist-backend 'nccl' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --dataset=cc3m  --val-dataset=cc3m \
    --exp_name='LLama_ret5_stack_b24' --image-dir='data/cc3m_byte/image_byte_224/'  --log-base-dir='runs' \
    --batch-size=48  --val-batch-size=64  --learning-rate=0.0003 --precision='bf16'  --print-freq=40 --max-len 35 --shared-emb-dim 768 \
    --qaprefix --LenRET=5 --n-visual-tokens=4  \
    --visual-model='openai/clip-vit-large-patch14' \
    --negbank \
    --Llama \
    --masktc \
    --allloss \
