from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, Repository

# Hugging Face Hub repo 名称（替换为你想要的）
repo_name = "icemoon28/qwen2.5-3b-finetuned"

# Hugging Face Hub 访问路径
repo_url = f"https://huggingface.co/{repo_name}"

# 你的 checkpoint 目录
checkpoint_path = "/root/autodl-tmp/runs/checkpoint-550"

# 加载模型 & tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# 推送到 Hugging Face Hub
model.push_to_hub(repo_name, max_shard_size="2GB")
tokenizer.push_to_hub(repo_name)

print(f"🚀 模型已上传至 {repo_url}")