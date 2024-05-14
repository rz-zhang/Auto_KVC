#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=2

# Base master port
base_port=25606

trap "exit" INT

# Outer loop for kv_compress_layer
for kv_layer in {11..31}; do
    # Loop through dim_compress values
    for dim in {256..512..128}; do
        torchrun --nproc_per_node=1 --master_port=$base_port \
            example_obqa.py \
            --ckpt_dir Meta-Llama-3-8B-Instruct/ \
            --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
            --max_seq_len 512 --max_batch_size 500 \
            --dim_compress $dim --kv_compress_layers $kv_layer
    done
done