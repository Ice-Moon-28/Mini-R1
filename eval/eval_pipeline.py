import os
from dataset.get_dataset import get_collect_fn, get_dataset
from reward.get_reward import get_reward
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
from torch.utils.data import DataLoader
from settings import device
from util.set_seed import set_seed

cache_dir = '/root/autodl-tmp/minir1'

def evaluate(model_name, dataset_name):
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    dataset = get_dataset(name=dataset_name, tokenizer=tokenizer)

    collect_fn = get_collect_fn(name=dataset_name, tokenizer=tokenizer)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collect_fn)

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
            output = model.generate(**prompt, max_length=1024)

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
            for reward_score in reward_scores:
                print(f"🏆 Reward: {reward_score[i]}")

            print("-" * 60)

    # 5️⃣ 计算平均奖励
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # 打印最终结果
    print("\n📊 Evaluation Completed!")
    print(f"✨ Average Reward: {avg_reward:.4f}")


def main():

    set_seed(seed=3047)
    # 参数配置
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    dataset_name = "guessing"

    # 调用评估函数
    evaluate(model_name, dataset_name)
