# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import os
import pickle
import torch

from llama import Dialog, Llama

def save_svd_results_for_all_layers(model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # 假设我们对Linear层进行处理
            weight_matrix = module.weight.data
            if weight_matrix.shape[0] > 1 and weight_matrix.shape[1] > 1:  # 只处理二维权重矩阵
                # 使用您的adaptive_svd_combined函数
                optimal_dim, *svd_results = adaptive_svd_combined(weight_matrix, weight_matrix)
                # 保存结果
                result_path = os.path.join(save_dir, f"{name}_svd.pkl")
                with open(result_path, 'wb') as f:
                    pickle.dump({
                        'optimal_dim': optimal_dim,
                        'svd_results': svd_results
                    }, f)
                print(f"Results saved for layer {name} with optimal dimension {optimal_dim}")


def main(
    ckpt_dir: str = 'Meta-Llama-3-8B-Instruct/',
    tokenizer_path: str = 'Meta-Llama-3-8B-Instruct/tokenizer.model',
    max_seq_len: int = 512,
    max_batch_size: int = 4,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )




if __name__ == "__main__":
    fire.Fire(main)
