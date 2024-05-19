#!/bin/bash
trap "exit" INT

# Base master port
base_port=25900

# Loop through dim_compress values
for dim in {80,64}; do
    torchrun --nproc_per_node 8 70b_obqa.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 16 --dim_compress $dim --kvc_config middle_40_layers
done