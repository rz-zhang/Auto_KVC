#!/bin/bash

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=3

# Base master port
base_port=25678

# Loop through dim_compress values
for dim in {640..1024..128}; do
    torchrun --nproc_per_node=1 --master_port=$base_port \
        example_xsum.py \
        --ckpt_dir Meta-Llama-3-8B-Instruct/ \
        --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
        --max_seq_len 2048 --max_batch_size 50 --dim_compress $dim

    # Increment the master port to avoid conflicts
    ((base_port++))
done

# The 'wait' command is no longer necessary since jobs are run sequentially
