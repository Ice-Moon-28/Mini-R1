from dataset.guess_word import GuessWordDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch
 
def get_countdown_chinese_dataset(tokenizer):
    dataset = load_dataset("parquet", data_files="dataset/countdown.parquet", split="train")

    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "你是一个乐于助人的助手。你会先在脑海中思考推理过程，然后再向用户提供答案。"
        },
        { 
            "role": "user",
            "content": f"使用数字 {numbers}，创建一个等式，使其结果等于 {target}。你可以使用基本的四则运算（+、-、*、/），每个数字只能使用一次。请在 <think> </think> 标签内展示你的推理过程，并在 <answer> </answer> 标签内返回最终的等式和答案，例如：<answer> (1 + 2) / 3 = 1 </answer>。"
        },
        {
            "role": "assistant",
            "content": "让我一步步推理解决这个问题。\n<think>"
        }]
        return {
            "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
            "target": target,
            "nums": numbers,
        }

    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    return dataset


def get_countdown_chinese_collate_fn(batch, tokenizer):
  
    prompts = [item['prompt'] for item in batch]
    targets = [item['target'] for item in batch]
    nums = [item['nums'] for item in batch]

    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    return {
        "prompts": {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
        "target": targets,
        "input": prompts,
        "nums": nums,
    }