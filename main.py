import os

import torch
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset.get_dataset import get_dataset
from reward.reward import equation_reward_func, format_reward_func
from huggingface_hub import login
import wandb
from settings import hf_token, wb_token

# Defined in the secrets tab in Google Colab
login(token=hf_token, add_to_git_credential=True)
wandb.login(key=wb_token)

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = '/root/autodl-tmp/minir1'


def train():

    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
        
    # our model we are going to use as policy 
    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",
        use_peft=True,
        load_in_4bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir=cache_dir)

    dataset = get_dataset(name='countdown', tokenizer=tokenizer)

    train_dataset, test_dataset = dataset.train_test_split(test_size=0.1)['train'], dataset.train_test_split(test_size=0.1)['test']

    # Hyperparameters
    training_args = GRPOConfig(
        output_dir="qwen-r1-aha-moment",
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        # GRPO specific parameters
        max_prompt_length=256,
        max_completion_length=1024, # max length of the generated output for our solution
        num_generations=3,
        beta=0.001,
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model_config.model_name_or_path,
        reward_funcs=[format_reward_func, equation_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

train()