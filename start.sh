poetry install
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="/root/autodl-tmp/minir1"
# poetry run python main.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
# accelerate launch --num_processes 3 --config_file deepspeed_zero3.yaml main.py --config receipes/grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
poetry run accelerate launch --num_processes 3 --config_file deepspeed_zero3.yaml main.py --config grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml