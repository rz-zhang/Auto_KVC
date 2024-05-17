

# OpenBook QA
torchrun --nproc_per_node 8 70b_obqa.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 10