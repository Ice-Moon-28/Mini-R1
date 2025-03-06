import math
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import torch


def get_gsm8k_dataset(tokenizer):
    dataset = load_dataset("openai/gsm8k", "main")['train']

    def generate_gsm8k_prompt(question, answer):
        answer_num = answer.split('####')[1].strip()

        prompt_template = [
            {
                "role": "system",
                "content": "You are a helpful and logical AI assistant specialized in solving math word problems. "
                        "Think through the problem carefully and provide a step-by-step solution."
            },
            {
                "role": "user",
                "content": f"{question}\n\nSolve this problem step by step, and return your reasoning inside <think> </think> tags. "
                        "Provide the final numerical answer inside <answer> </answer> tags, for example:\n"
                        "<answer> 42 </answer>."
            },
            {
                "role": "assistant",
                "content": "Let's solve this step by step.\n<think>"
            }
        ]

        return {
            "question": question,
            "prompt": tokenizer.apply_chat_template(prompt_template, tokenize=False, continue_final_message=True),
            "target": answer_num,
        }

    original_columns = dataset.column_names

    dataset = dataset.map(
        lambda x: generate_gsm8k_prompt(x["question"], x["answer"]),
        remove_columns=original_columns,
    )

    return dataset


def get_gsm8k_collate_fn(batch, tokenizer):
  
    prompts = [item['prompt'] for item in batch]
    targets = [item['target'] for item in batch]
    questions = [item["question"] for item in batch]


    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    return {
        "prompts": {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
        "target": targets,
        "questions": questions,
        "input": prompts,
    }

if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = get_gsm8k_dataset(tokenizer=tokenizer)