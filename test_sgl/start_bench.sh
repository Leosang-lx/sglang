#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PREFIX="/home/liux/big_file/"
MODELID="Qwen/Qwen3-8B"

# python -m sglang.bench_offline_throughput \
#     --model-path /home/liux/big_file/Qwen/Qwen3-8B/ \
#     --dataset-path /home/liux/big_file/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --attention-backend flashinfer \

python -m sglang.bench_one_batch \
    --model-path $PREFIX$MODELID \
    --custom-batch \