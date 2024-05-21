'''
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port 25788 gsm8k.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 10 --dim_compress 128 --kvc_config second_half_7b

CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node 2 --master_port 25333 gsm8k.py \
   --ckpt_dir /localscratch/rongzhi/kvcache/llama/llama-2-inst-8b/ \
    --tokenizer_path /localscratch/rongzhi/kvcache/llama/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 20 --dim_compress 256 --kvc_config second_half_7b
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

BASELINE = []
SECPMD_HALF_7B = list(range(16, 32))
SECOND_HALF_13B = list(range(20,40))
ALL_LAYERS = list(range(40))
CUSTOM_LAYERS = list(range(15,35))
LAST_15_LAYERS = list(range(25,40))

DATA_SLICE = 100

KVC_CONFIG_DICT = {
    'all_layers': ALL_LAYERS,
    'second_half_7b': SECPMD_HALF_7B,
    'second_half_13b': SECOND_HALF_13B,
    'baseline': BASELINE,
    'custom_layers': CUSTOM_LAYERS,
    'last_15_layers': LAST_15_LAYERS,
}

def create_prompts_from_data(data, examples):
    content = f"Please give a step-by-step answer to the question. You have to put your final numeric answer at the end, without any extra sign, prefix, or suffix, just pure integer numbers, in the format: \n#### answer\n Done, make sure to separate the final numeric answer with \n####"
    system_message = 'You are a Math teacher who is capable of generating high-quality math solutions. Here are some examples:'

    prompts = []
    references = []

    example_section = ""
    for ex_question, ex_answer in examples:
        example_section += f"\nExample Question: {ex_question}\nExample Answer: {ex_answer}\n"

    for question, answer in zip(data['question'], data['answer']):
        prompt = f"{example_section}\nQuestion: {question}\n{content}."
        prompts.append(prompt)
        _, extracted_answer = extract_answer(answer)
        references.append(extracted_answer)
    return prompts, references

def extract_answer(completion):
    start_idx = completion.find("####")
    if start_idx == -1:
        return completion, 'None'
    start_idx += 4  # Move past '####'
    end_idx = completion.find('\n', start_idx)
    if end_idx == -1:
        end_idx = len(completion)
    answer = completion[start_idx:end_idx].strip()
    return completion[:end_idx], answer


def calculate_accuracy(predictions, references):
    correct = sum([1 for (_, pred), ref in zip(predictions, references) if pred.lower() == ref.lower()])
    return correct, len(predictions)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 1024,
    max_gen_len: int = 1024,
    max_batch_size: int = 10,
    dim_compress: int = 4096,
    kvc_config: str = 'baseline',
    kv_compress_layers: Optional[List[int]] = None,
    adaptive: bool = False,
    num_shot: int = 1,
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

    # Load the BoolQ dataset
    train_dataset = load_dataset("gsm8k", 'main', split='train')
    validation_dataset = load_dataset("gsm8k", 'main', split='test')

    if DATA_SLICE:
        train_dataset = train_dataset.select(range(DATA_SLICE))
        validation_dataset = validation_dataset.select(range(DATA_SLICE))

    example_section = [(train_dataset[i]['question'], train_dataset[i]['answer']) for i in range(num_shot)]

    # Generate prompts and reference answers using the modified function
    prompts, reference_answers = create_prompts_from_data(validation_dataset, example_section)

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
        print(batch_predictions)
        print(batch_references)
        predictions.extend(batch_predictions)

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
    filename = f"/localscratch/rongzhi/kvcache/llama/eval/gsm8k/llama-3-inst-8b/{kvc_config}_dim_{dim_compress}_{timestamp}.json"
    # Check if the directory exists, and if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved in {filename}")


if __name__ == "__main__":
    fire.Fire(main)
