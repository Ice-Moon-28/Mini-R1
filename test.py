import os
from dataset.get_dataset import get_collect_fn, get_dataset
from model.get_model import get_model
from reward.get_reward import get_reward
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
from torch.utils.data import DataLoader
from settings import device
from util.set_seed import set_seed
from trl import GRPOConfig
from transformers.trainer_utils import get_last_checkpoint

import torch
from transformers import AutoModel, AutoTokenizer
dataset_name = 'guessing'
# ✅ 指定模型目录
checkpoint_path = "/root/autodl-tmp/runs/checkpoint-600"

# ✅ 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

tokenizer.pad_token_id = tokenizer.eos_token_id

# ✅ 加载模型（Hugging Face 会自动处理分片）
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)


model = model.cuda()
model.eval()

dataset = get_dataset(name=dataset_name, tokenizer=tokenizer)

collect_fn = get_collect_fn(name=dataset_name, tokenizer=tokenizer)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collect_fn)

reward_fns = get_reward(name=dataset_name)

predictions, targets, rewards = [], [], []
print("🔍 Starting Evaluation...")

for batch in tqdm(dataloader, desc="Evaluating Samples"):

    original_prompt = batch["input"]

    prompt = {
        "input_ids": batch["prompts"]["input_ids"].to(device),
        "attention_mask": batch["prompts"]["attention_mask"].to(device)
    }
    target = batch["target"]

    # Tokenize and generate prediction
    with torch.no_grad():
        output = model.generate(**prompt, max_new_tokens=768)

    total_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    pure_completions = [pred[len(prompt_text):] for pred, prompt_text in zip(output, prompt['input_ids'])]

    prediction = tokenizer.batch_decode(pure_completions, skip_special_tokens=True)

    reward_scores = [fn(prediction, target) for fn in reward_fns]

    rewards.extend(reward_scores)
    
    for i in range(len(original_prompt)):
        print("-" * 60)
        total_output_i= total_output[i]
        answer_i = target[i]
        print(f"🤖 Output: {total_output_i}")
        print(f"🎯 Target: {answer_i}")
        for j in range(len(reward_scores)):
            reward_score = reward_scores[j]
            print(f"🏆 Reward{j}: {reward_score[i]}")

        print("-" * 60)

# 5️⃣ 计算平均奖励
avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

# 打印最终结果
print("\n📊 Evaluation Completed!")
print(f"✨ Average Reward: {avg_reward:.4f}")
