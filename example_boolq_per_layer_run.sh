#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=3

# Base master port
base_port=21014

trap "exit" INT

for kv_layer in {3,9,18,24,26,27,28,29,30,31}; do
    for dim in {256,384,512}; do
        torchrun --nproc_per_node=1 --master_port=$base_port \
            example_boolq.py \
            --ckpt_dir Meta-Llama-3-8B-Instruct/ \
            --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
            --max_seq_len 2048 --max_batch_size 20 \
            --dim_compress $dim --kv_compress_layers $kv_layer &
        base_port=$((base_port + 1))  # Increment the base port for the next job
    done
    wait  # Wait for all jobs in this batch to finish before starting the next batch
done

# The 'wait' command is no longer necessary since jobs are run sequentially
