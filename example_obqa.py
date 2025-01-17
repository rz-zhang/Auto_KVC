'''
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node 1 --master_port 25677 example_obqa.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 3072 --max_batch_size 20 --dim_compress 512 --kvc_config baseline

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 25777 example_obqa.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 3072 --max_batch_size 20 --dim_compress 1024 --dim_compress_v 512 --kvc_config second_half_layers

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 25677 example_obqa.py \
--ckpt_dir Meta-Llama-3-8B-Instruct/ \
--tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
--max_seq_len 3072 --max_batch_size 20 --dim_compress 512 --kvc_config baseline --config_file config/llama3-8b-anneal-4.txt
'''

from typing import List, Optional

import fire
import os
from datasets import load_dataset
import re
from tqdm import tqdm
import json
import datetime

from llama import Dialog, Llama

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
    answers = []
    for i in range(len(data['id'])):
        question = data['question_stem'][i]
        choices = data['choices'][i]['text']
        labels = data['choices'][i]['label']
        formatted_choices = "\n".join(f"{label}) {text}" for label, text in zip(labels, choices))
        prompt = f"Question: {question}\nOptions:\n{formatted_choices}\nAnswer:"
        prompts.append(prompt)
        answers.append(data['answerKey'][i])
    return prompts, answers

def extract_answer_labels(outputs):
    answer_labels = []
    # Regex pattern to capture answer labels right after "Answer: " followed by a single character (A-D)
    pattern = r'Answer:\s*([A-D])'
    for output in outputs:
        match = re.search(pattern, output)
        if match:
            # Append the first capturing group which corresponds to the answer label
            answer_labels.append(match.group(1))
        else:
            # Append None if no match is found
            answer_labels.append(None)
    return answer_labels

def extract_option_label(outputs):
    answer_labels = []
    for output in outputs:
        match = re.search(r'\b([A-D])\b', output)
        if match:
            answer_labels.append(match.group(1))
        else:
            answer_labels.append(None)
    return answer_labels

def calculate_accuracy(predictions, correct_answers):
    total = len(predictions)
    correct = sum(1 for pred, answer in zip(predictions, correct_answers) if pred == answer)
    accuracy = correct / total
    return accuracy

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    max_gen_len: int = 64,
    max_batch_size: int = 10,
    dim_compress: int = 1024,
    kv_compress_layers: int = 0,
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
        kv_compress_layers=kv_compress_layers,
        adaptive=adaptive,
        custom_kvc_config=custom_kvc_config,
        dim_compress_v=dim_compress_v,
    )

    model_stats = generator.get_model_stats()
    model_stats_seperate = generator.get_model_stats_seperate()

    dataset = load_dataset("allenai/openbookqa", split='test')
    prompts, correct_answers = create_prompts_from_data(dataset)

    predictions = []
    extracted_answers = []
    correct_count = 0

    # Processing prompts in batches
    num_prompts = len(prompts)
    for i in tqdm(range(0, num_prompts, max_batch_size), desc="Generating completions"):
        batch_prompts = prompts[i:i + max_batch_size]
        results = generator.text_completion(
            batch_prompts,
            max_gen_len=max_gen_len,
            temperature=0.1,
            top_p=0.5,
        )

        # Iterate over results to extract answers and calculate accuracy
        for j, result in enumerate(results):
            generated_text = result['generation'].strip()
            predictions.append(generated_text)

            extracted_answer = extract_option_label([generated_text])[0]
            extracted_answers.append(extracted_answer)

            # Update accuracy if the answer is correct
            if extracted_answer == correct_answers[i + j]:
                correct_count += 1

            # Optionally print each prediction and its extracted answer
            # print(f"Prompt: {batch_prompts[j]}")
            # print(f"Generated: {generated_text}")
            # print(f"Extracted Answer: {extracted_answer}")

        # Display the current accuracy after processing each batch
        accuracy = correct_count / len(extracted_answers)
        print(f"Current Accuracy: {accuracy * 100:.2f}%")

    # Print final results
    # print("\nFinal Predictions:", predictions)
    # print("Extracted Answers:", extracted_answers)
    # print("Correct Answers:", correct_answers)
    # print(f"Final Accuracy: {accuracy * 100:.2f}%")

    results = {
        "accuracy": accuracy,
        "model_stats": model_stats,
        "model_states_seperate": model_stats_seperate,
        "dim_compress": dim_compress,
        "kv_compress_layers": kv_compress_layers,
        "predictions": predictions,
        "extracted_answers": extracted_answers,
        "correct_answers": correct_answers
    }

    # Prepare output file name
    if not isinstance(kv_compress_layers, list):
        kv_compress_layers = [kv_compress_layers]
    kv_compress_layers_str = '-'.join(map(str, kv_compress_layers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    if adaptive:
        filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa_adaptive_{timestamp}.json"
    else:
        filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa/llama-3-inst-8b-res/layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa/llama-3-inst-8b/all_layer_dim_{dim_compress}_{timestamp}.json"
    # Check if the directory exists, and if not, create it
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved in {filename}")
if __name__ == "__main__":
    fire.Fire(main)