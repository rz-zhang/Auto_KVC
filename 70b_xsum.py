'''
torchrun --nproc_per_node 8 70b_xsum.py \
--ckpt_dir Meta-Llama-3-70B-Instruct/ \
--tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
--max_seq_len 2048 --max_batch_size  --max_gen_len 256
'''

from typing import List, Optional

import fire
import os
from datasets import load_dataset
import re
from tqdm import tqdm
import json
import datetime
import numpy as np

from llama import Dialog, Llama
from evaluate import load

LAYER_MAPPING = {i: [i] for i in range(32)}
DIM_256_ROUGE_1_RANK = [
    6, 9, 2, 20, 26, 3, 28, 8, 14, 17, 22,
    1, 29, 30, 16, 18, 21, 23, 19, 5, 10, 4, 24,
    25, 7, 11, 12, 27, 15, 31, 13, 0]

AVE_DIM_128_256_384_512_ROUGE_1_RANK = [
    6, 20, 9, 26, 1, 18, 29, 17, 2, 23, 3, 8,
    30, 22, 12, 15, 28, 14, 5, 24, 19, 21, 25,
    10, 7, 16, 27, 11, 31, 13, 4, 0]

# Exclude layers 1-9 from AVE_DIM_128_256_384_512_ROUGE_1_RANK
CUSTOM_LAYERS = [20, 26, 18, 29, 17, 23,
    30, 22, 12, 15, 28, 14, 24, 19, 21, 25,
    10, 16, 27, 11, 31, 13, 4, 0]

SECOND_HALF_LAYERS = list(range(40, 80))
LAST_20_LAYERS = list(range(60, 80))
ALL_LAYERS = list(range(80))
MIDDLE_60_LAYERS = list(range(10, 70))
Middle_40_LAYERS = list(range(20, 60))
LAST_60_LAYERS = list(range(20, 80))
BASELINE = []
CUSTOM_LAYERS = list(range(40,75))

KVC_CONFIG_DICT = {
    'all_layers': ALL_LAYERS,
    'second_half': SECOND_HALF_LAYERS,
    'last_20_layers': LAST_20_LAYERS,
    'middle_60_layers': MIDDLE_60_LAYERS,
    'last_60_layers': LAST_60_LAYERS,
    'middle_40_layers': Middle_40_LAYERS,
    'baseline': BASELINE,
    'custom_layers': CUSTOM_LAYERS,
}


# kv_compress_layers=LAYER_MAPPING.get(kv_compress_layers, [])

def create_prompts_from_data(data):
    prompts = []
    references = []
    for article, summary in zip(data['document'], data['summary']):
        prompt = f"Provide a concise summary of the text below: {article}\n\nSummary:"
        prompts.append(prompt)
        references.append(summary)
    return prompts, references

def calculate_rouge_scores(generated_summaries, reference_summaries):
    rouge = load("rouge")
    scores = rouge.compute(predictions=generated_summaries, references=reference_summaries)
    return scores

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_gen_len: int = 64,
    max_batch_size: int = 10,
    dim_compress: int = 1024,
    kvc_config: str = 'baseline',
    adaptive: bool = False,
):
    kv_compress_layers = KVC_CONFIG_DICT[kvc_config]
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dim_compress=dim_compress,
        kv_compress_layers=kv_compress_layers,
    )

    model_stats = generator.get_model_stats()

    dataset = load_dataset("xsum", split='test')
    prompts, reference_summaries = create_prompts_from_data(dataset[:100])


    predictions = []
    num_prompts = len(prompts)
    for i in tqdm(range(0, num_prompts, max_batch_size), desc="Generating summaries"):
        batch_prompts = prompts[i:i + max_batch_size]
        results = generator.text_completion(
            batch_prompts,
            max_gen_len=max_gen_len,
            temperature=0.6,
            top_p=0.9,
        )
        for result in results:
            predictions.append(result['generation'].strip())

        if predictions:
            accumulated_rouge_scores = calculate_rouge_scores(predictions, reference_summaries[:len(predictions)])
            print(f"Accumulated ROUGE scores after {len(predictions)} prompts: {accumulated_rouge_scores}")

    # 计算ROUGE得分
    rouge_scores = calculate_rouge_scores(predictions, reference_summaries)

    print("ROUGE Scores:", rouge_scores)
    print('Compression Ratio', np.mean(model_stats['compression_ratio']))

    results = {
        "rouge_scores": rouge_scores,
        "model_stats": model_stats,
        "dim_compress": dim_compress,
        "kv_compress_layers": kv_compress_layers,
        "Num of predictions": len(predictions),
        "predictions": predictions,
        "reference_summaries": reference_summaries,
    }

    # Prepare output file name
    if not isinstance(kv_compress_layers, list):
        kv_compress_layers = [kv_compress_layers]
    kv_compress_layers_str = '-'.join(map(str, kv_compress_layers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    if adaptive:
        filename = f"/llama3/eval/xsum_per_layer_kvc/adaptive_{timestamp}.json"
    else:
        filename = f"/llama3/eval/xsum_per_layer_kvc/layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    #filename = f"/localscratch/rongzhi/kvcache/llama3/eval/xsum/test1k/custom_layers_rouge1_top16_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    # filename = f"/localscratch/rongzhi/kvcache/llama3/eval/xsum/test1k/ave_dim_128_256_384_512_rouge1_top16_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    #filename = f"./eval/xsum/test1k/{kvc_config}_layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    filename = f"~/mycontainer/rongzhi/KVC/eval/xsum/test1k/{kvc_config}_dim_{dim_compress}_{timestamp}.json"
    directory = os.path.dirname(filename)
    try:
        # Create the directory, ignore if it already exists
        os.makedirs(directory, exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(results, file, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Results saved in {filename}")
if __name__ == "__main__":
    fire.Fire(main)