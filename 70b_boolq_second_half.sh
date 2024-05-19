#!/bin/bash
trap "exit" INT

# Base master port
base_port=28800

# Loop through dim_compress values
for dim in {2048,1536,1024,512}; do
    torchrun --nproc_per_node 8 70b_boolq.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 3072 --max_batch_size 80 --dim_compress $dim --kvc_config second_half
done