from dataset.guess_word import GuessWordDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os
 
def get_countdown_dataset(tokenizer):
    dataset = load_dataset("parquet", data_files="dataset/countdown.parquet", split="train")

    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
        { 
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    return dataset

