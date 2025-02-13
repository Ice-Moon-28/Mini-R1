from transformers import AutoTokenizer, AutoModelForCausalLM


def get_Qwen_2_5_3B(model_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct',
        **model_config,
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

    return model, tokenizer


def get_model(
    model_name = "teknium/OpenHermes-2.5-Mistral-7B",
    model_config=None,
):
    if model_name == 'Qwen/Qwen2.5-3B-Instruct':
        model, tokenizer = get_Qwen_2_5_3B(model_config)
        return model, tokenizer
    
    else:
        return None, None
