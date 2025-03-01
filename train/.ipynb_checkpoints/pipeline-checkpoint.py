import os

import torch
from reward.get_reward import get_reward
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from dataset.get_dataset import get_dataset
from reward.count_down_reward import equation_reward_func, format_reward_func
from huggingface_hub import login
import wandb
from settings import hf_token, wb_token
from transformers.trainer_utils import get_last_checkpoint
import logging
from dataclasses import dataclass


# @dataclass
# class ScriptArguments:
#     dataset_name: str = "countdown"
#     tokenizer_name_or_path: str = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


# # Defined in the secrets tab in Google Colab
login(token=hf_token, add_to_git_credential=True)
wandb.login(key=wb_token)

import os

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cache_dir = '/root/autodl-tmp/minir1'


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


import os

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

cache_dir = '/root/autodl-tmp/minir1'

@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "Jiayi-Pan/Countdown-Tasks-3to4"

def train(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):

    os.makedirs(cache_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
        
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            print(args, state, control)
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir=cache_dir)

    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    
    dataset = get_dataset(name=script_args.dataset_id_or_path, tokenizer=tokenizer)
    # dataset = dataset.filter(lambda example: example['label'] == 'association')

    split_dataset = dataset.train_test_split(test_size=0.25)

    train_dataset, eval_dataset = split_dataset['train'], split_dataset['test']

    rewards_fn = get_reward(name=script_args.dataset_id_or_path)

    training_args.save_total_limit = 1

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=rewards_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[PrinterCallback()]
    )

    print("LoRA Config:", get_peft_config(model_args))

    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    else:
        train_result = trainer.train()

        eval_res = trainer.evaluate()

        print(eval_res)

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})

        # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

 


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))

    model_args, script_args, training_args = parser.parse_args_and_config()

    training_args.max_grad_norm = 1.0

    # Run the main training loop
    train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()