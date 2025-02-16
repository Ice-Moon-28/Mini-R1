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

# def train(
#     model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
# ):

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype="bfloat16",
#         bnb_4bit_use_double_quant=True,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         quantization_config=quantization_config,
#         torch_dtype="bfloat16",
#         # attn_implementation="flash_attention_2",
#     )

#     model.config.use_cache = False

#     os.makedirs(cache_dir, exist_ok=True)

#     os.environ["HF_HOME"] = cache_dir
        
#     logger.info(f"Model parameters {model_args}")
#     logger.info(f"Training/evaluation parameters {training_args}")


#     tokenizer = AutoTokenizer.from_pretrained(
#         (
#             script_args.tokenizer_name_or_path
#             if script_args.tokenizer_name_or_path
#             else model_args.model_name_or_path
#         ),
#         revision=model_args.model_revision,
#         trust_remote_code=model_args.trust_remote_code,
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token


#     ###############
#     # Load datasets
#     ###############
#     # Load dataset from Hugging Face Hub
#     dataset = get_dataset(script_args.dataset_name, tokenizer=tokenizer)
#     # select a random subset of 50k samples
#     dataset = dataset.shuffle(seed=42).select(range(50000))

#     dataset = get_dataset(name='countdown', tokenizer=tokenizer)

#     train_dataset, test_dataset = dataset.train_test_split(test_size=0.1)['train'], dataset.train_test_split(test_size=0.1)['test']
#     training_args.model_init_kwargs = {"use_cache": False}

#     trainer = GRPOTrainer(
#         model=model,
#         reward_funcs=[format_reward_func, equation_reward_func],
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         peft_config=get_peft_config(model_args),
#     )

#     last_checkpoint = get_checkpoint(training_args)
#     if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
#         logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

#     # Train the model
#     logger.info(
#         f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
#     )

#     if last_checkpoint is not None:
#         train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

#     else:
#         train_result = trainer.train()

#     # Log and save metrics
#     metrics = train_result.metrics
#     metrics["train_samples"] = len(train_dataset)
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)
#     trainer.save_state()

#     logger.info("*** Training complete ***")

#     ##################################
#     # Save model and create model card
#     ##################################

#     logger.info("*** Save model ***")
#     trainer.model.config.use_cache = True
#     trainer.save_model(training_args.output_dir)
#     logger.info(f"Model saved to {training_args.output_dir}")
#     training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

#     tokenizer.save_pretrained(training_args.output_dir)
#     logger.info(f"Tokenizer saved to {training_args.output_dir}")

#     # Save everything else on main process
#     if trainer.accelerator.is_main_process:
#         trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
#     # push to hub if needed
#     if training_args.push_to_hub is True:
#         logger.info("Pushing to hub...")
#         trainer.push_to_hub()

#     logger.info("*** Training complete! ***")


# def eval():
#     cache_dir = "/root/autodl-tmp/minir1"  # 确保 cache 目录正确
#     checkpoint_dir = "qwen-r1-aha-moment/checkpoint-100"  # 这里可以换成最新的 checkpoint
#     model_name = "Qwen/Qwen2.5-3B-Instruct"

#     # 加载 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # **先加载基础模型**
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_name,  # 加载原始模型
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )

#     # **再加载 PEFT 适配的权重**
#     model = PeftModel.from_pretrained(base_model, checkpoint_dir)

#     # 确保模型在 GPU 上
#     model = model.to("cuda" if torch.cuda.is_available() else "cpu")

 

#     # 选择测试集
#     dataset = get_dataset(name='countdown', tokenizer=tokenizer)
#     test_dataset = dataset.train_test_split(test_size=0.1)['test']

#     # 遍历测试集并生成输出
#     for item in test_dataset:
#         sample = item["prompt"]  
#         print("\n[INPUT QUESTION]:", sample)

#         input_ids = tokenizer(sample, return_tensors="pt").input_ids.to(model.device)
#         with torch.no_grad():
#             output_ids = model.generate(input_ids, max_length=768, num_return_sequences=1)

#         output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         print("\n[MODEL COMPLETION]:", output_text)

# # eval()

# if __name__ == '__main__':
#     parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
#     model_args, script_args, training_args = parser.parse_args_and_config()

#     # Run the main training loop
#     train(model_args, script_args, training_args)

import os

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

    dataset = get_dataset(name=script_args.dataset_id_or_path, tokenizer=tokenizer)

    split_dataset = dataset.train_test_split(test_size=0.2)

    train_dataset, test_dataset = split_dataset['train'], split_dataset['test']

    rewards_fn = get_reward(name=script_args.dataset_id_or_path)

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=rewards_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=[PrinterCallback()]
    )

    trainer.train()


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()