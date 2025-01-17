#!/bin/bash
trap "exit" INT

# Base master port
base_port=28800

# Loop through dim_compress values
for dim in {112,96,80,64,32,16,8}; do
    torchrun --nproc_per_node 8 70b_gsm8k.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_batch_size 16 --dim_compress $dim --kvc_config last_20_layers
done