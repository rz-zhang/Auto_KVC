'''
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --master_port 25614 non_consecutive_kvc_obqa.py \
    --ckpt_dir Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 512 --max_batch_size 10 --dim_compress 512
'''

from typing import List, Optional

import fire
from datasets import load_dataset
import re
from tqdm import tqdm
import json
import datetime
import numpy as np

from llama import Dialog, Llama

LAYER_MAPPING = {i: [i] for i in range(32)}
AVE_DIM_256_384_512_LAYER_RANK = [
    15, 19, 23, 24, 25, 27, 28, 29, 18, 30, 22, 21,
    20, 31, 26, 17, 13, 14, 16, 12, 1, 9, 11, 7,
    8, 2, 10, 6, 5, 4, 3, 0]
# kv_compress_layers=LAYER_MAPPING.get(kv_compress_layers, [])

KVC_CONFIG_1 = [
    1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
    896, 896, 896, 896, 896, 896, 896, 896,
    256, 256, 256, 256, 256, 256, 256, 256,
    224, 224, 224, 224, 224, 224, 224, 224,
]

KVC_CONFIG_2 = [
    1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
    512, 512, 512, 512, 512, 512, 512, 512,
    256, 256, 256, 256, 256, 256, 256, 256,
    192, 192, 192, 192, 192, 192, 192, 192,
]

SECOND_HALF_LAYERS = list(range(16, 32))
LAST_LAYERS = list(range(20, 32))
BASELINE = []

KVC_CONFIG_DICT = {
    'kvc_1': KVC_CONFIG_1,
    'kvc_2': KVC_CONFIG_2,
    'second_half': SECOND_HALF_LAYERS,
    'last_layers': LAST_LAYERS,
    'baseline': BASELINE,
}

DATA_SLICE = None

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

# def extract_answer_labels(outputs):
#     answer_labels = []
#     # Regex pattern to capture answer labels right after "Answer: " followed by a single character (A-D)
#     pattern = r'Answer:\s*([A-D])'
#     for output in outputs:
#         match = re.search(pattern, output)
#         if match:
#             # Append the first capturing group which corresponds to the answer label
#             answer_labels.append(match.group(1))
#         else:
#             # Append None if no match is found
#             answer_labels.append(None)
#     return answer_labels

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
    kvc_config: str = 'baseline',
):
    kv_compress_layers = KVC_CONFIG_DICT[kvc_config]
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        dim_compress=dim_compress,
        kv_compress_layers=kv_compress_layers,
        custom_kvc_config=KVC_CONFIG_2,
    )

    model_stats = generator.get_model_stats()

    dataset = load_dataset("allenai/openbookqa", split='test')
    if DATA_SLICE:
        dataset = dataset[:DATA_SLICE]
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
        "dim_compress": dim_compress,
        "kv_compress_layers": kv_compress_layers,
        "predictions": predictions,
        "extracted_answers": extracted_answers,
        "correct_answers": correct_answers
    }
    print('Compression Ratio', np.mean(model_stats['compression_ratio']))

    # Prepare output file name
    if not isinstance(kv_compress_layers, list):
        kv_compress_layers = [kv_compress_layers]
    kv_compress_layers_str = '-'.join(map(str, kv_compress_layers))

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")

    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa_layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa/ave_dim_256_384_512_top16_layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa/custom_config_test_{timestamp}.json"
    filename = f"/localscratch/rongzhi/kvcache/llama3/eval/obqa/{kvc_config}_layer_{kv_compress_layers_str}_dim_{dim_compress}_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved in {filename}")
if __name__ == "__main__":
    fire.Fire(main)