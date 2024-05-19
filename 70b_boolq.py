'''
torchrun --nproc_per_node 8 70b_boolq.py \
    --ckpt_dir Meta-Llama-3-70B-Instruct/ \
    --tokenizer_path Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_batch_size 10
'''

from datasets import load_dataset
import os
import fire
from tqdm import tqdm
import json
import datetime
from evaluate import load
from typing import List, Optional

from llama import Llama

SECOND_HALF_LAYERS = list(range(40, 80))
LAST_20_LAYERS = list(range(60, 80))
ALL_LAYERS = list(range(80))
MIDDLE_60_LAYERS = list(range(10, 70))
Middle_40_LAYERS = list(range(20, 60))
LAST_60_LAYERS = list(range(20, 80))
BASELINE = []

KVC_CONFIG_DICT = {
    'all_layers': ALL_LAYERS,
    'second_half': SECOND_HALF_LAYERS,
    'last_20_layers': LAST_20_LAYERS,
    'middle_60_layers': MIDDLE_60_LAYERS,
    'last_60_layers': LAST_60_LAYERS,
    'middle_40_layers': Middle_40_LAYERS,
    'baseline': BASELINE,
}
def create_prompts_from_data(data):
    prompts = []
    references = []
    for question, passage, answer in zip(data['question'], data['passage'], data['answer']):
        prompt = f"Passage: {passage}\nQuestion: {question}\n\nAnswer (yes or no):"
        prompts.append(prompt)
        references.append("yes" if answer else "no")
    return prompts, references

def extract_answer(generated_text: str) -> str:
    normalized_text = generated_text.lower().strip()
    if normalized_text.startswith("yes"):
        return "yes"
    elif normalized_text.startswith("no"):
        return "no"
    return "unknown"


def calculate_accuracy(predictions, references):
    correct = sum([1 for pred, ref in zip(predictions, references) if pred.lower() == ref.lower()])
    return correct, len(predictions)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
    max_gen_len: int = 32,
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
        adaptive=adaptive,
    )

    dataset = load_dataset("boolq", split='validation')
    prompts, reference_answers = create_prompts_from_data(dataset[:1000])

    predictions = []
    total_correct = 0
    total_samples = 0
    num_prompts = len(prompts)
    for i in tqdm(range(0, num_prompts, max_batch_size), desc="Generating answers"):
        batch_prompts = prompts[i:i + max_batch_size]
        batch_references = reference_answers[i:i + max_batch_size]
        results = generator.text_completion(
            batch_prompts,
            max_gen_len=max_gen_len,
            temperature=0.6,
            top_p=0.9,
        )
        batch_predictions = [extract_answer(result['generation'].strip()) for result in results]
        predictions.extend(batch_predictions)

        # Update cumulative accuracy
        batch_correct, batch_count = calculate_accuracy(batch_predictions, batch_references)
        total_correct += batch_correct
        total_samples += batch_count
        cumulative_accuracy = total_correct / total_samples
        print(f"Cumulative accuracy after processing {total_samples} samples: {cumulative_accuracy:.2%}")

        # Final accuracy
    final_accuracy = total_correct / total_samples
    print("Final Accuracy:", final_accuracy)

    model_stats = generator.get_model_stats()

    # Save results
    results = {
        "final_accuracy": final_accuracy,
        "model_stats": model_stats,
        "dim_compress": dim_compress,
        "kv_compress_layers": kv_compress_layers,
        "predictions": predictions,
        "reference_answers": reference_answers,
    }
    if not isinstance(kv_compress_layers, list):
        kv_compress_layers = [kv_compress_layers]
    kv_compress_layers_str = '-'.join(map(str, kv_compress_layers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/boolq_test_1k_dim_{dim_compress}_{timestamp}.json"
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/boolq/baseline_all_data_{timestamp}.json"
    filename = f"~/mycontainer/rongzhi/KVC/eval/boolq/{kvc_config}_layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    filename = f"~/mycontainer/rongzhi/KVC/eval/boolq/{kvc_config}_dim_{dim_compress}_{timestamp}.json"
    # Check if the directory exists, and if not, create it
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
