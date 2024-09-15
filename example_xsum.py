'''
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 25688 example_xsum.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_batch_size 50 --dim_compress 256 -- max_gen_len 512

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 25644 example_xsum.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_batch_size 50 --dim_compress 256  --kvc_config baseline --config_file config/llama3-8b-anneal-min16-max64-protect16.txt
'''

from datasets import load_dataset
import fire
from tqdm import tqdm
import json
import datetime
from evaluate import load
from typing import List, Optional

from llama import Llama

ALL_LAYERS = list(range(32))
SECOND_HALF_LAYERS = list(range(16, 32))
LAST_LAYERS = list(range(12, 32))
LAYER_0 = [0]
SHALLOW_BLOCKS = [0,4]
BASELINE = []

KVC_CONFIG_DICT = {
    "all_layers": ALL_LAYERS,
    "second_half_layers": SECOND_HALF_LAYERS,
    "last_layers": LAST_LAYERS,
    "layer_0": LAYER_0,
    "shallow_blocks": SHALLOW_BLOCKS,
    "baseline": BASELINE,
}

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
    max_seq_len: int = 2048,
    max_gen_len: int = 128,
    max_batch_size: int = 10,
    dim_compress: int = 1024,
    kv_compress_layers: Optional[List[int]] = None,
    adaptive: bool = False,
    kvc_config: str = 'baseline',
    config_file: Optional[str] = None,
    dim_compress_v: Optional[int] = None,
):
    if config_file is not None:
        with open(config_file, 'r') as file:
            custom_kvc_config = [int(line.strip()) for line in file]
    else:
        custom_kvc_config = None
    kv_compress_layers = KVC_CONFIG_DICT[kvc_config]
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dim_compress=dim_compress,
        kv_compress_layers=BASELINE,
        adaptive=adaptive,
        custom_kvc_config=custom_kvc_config,
        dim_compress_v=dim_compress_v,
    )

    model_stats = generator.get_model_stats()
    model_stats_seperate = generator.get_model_stats_seperate()

    dataset = load_dataset("xsum", split='test')
    prompts, reference_summaries = create_prompts_from_data(dataset[:200])

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

    # 计算ROUGE得分
    rouge_scores = calculate_rouge_scores(predictions, reference_summaries)

    # 输出ROUGE得分
    print("ROUGE Scores:", rouge_scores)

    # 保存结果
    results = {
        "rouge_scores": rouge_scores,
        "model_states_seperate": model_stats_seperate,
        "predictions": predictions,
        "reference_summaries": reference_summaries,
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/xsum_test_1k_dim_{config_file}_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved in {filename}")

if __name__ == "__main__":
    fire.Fire(main)
