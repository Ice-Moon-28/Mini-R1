from transformers import AutoTokenizer, AutoModelForCausalLM


def get_Qwen_2_5_3B_instruct(model_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B-Instruct',
        **model_config,
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return model, tokenizer

def get_Qwen_2_5_7B_instruct(model_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct',
        **model_config,
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_Qwen_2_5_3B(model_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B',
        **model_config,
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')

    tokenizer.padding_side = "left"
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_Qwen_2_5_7B(model_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B',
        **model_config,
    )

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')

    tokenizer.padding_side = "left"
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_model(
    model_name = "teknium/OpenHermes-2.5-Mistral-7B",
    model_config=None,
):
    if model_name == 'Qwen/Qwen2.5-3B-Instruct':
        model, tokenizer = get_Qwen_2_5_3B_instruct(model_config)
        return model, tokenizer
    elif model_name == 'Qwen/Qwen2.5-3B':
        model, tokenizer = get_Qwen_2_5_3B(model_config)
        return model, tokenizer

    elif model_name == 'Qwen/Qwen2.5-7B-Instruct':
        model, tokenizer = get_Qwen_2_5_7B_instruct(model_config)
        return model, tokenizer

    elif model_name == 'Qwen/Qwen2.5-7B':
        model, tokenizer = get_Qwen_2_5_7B(model_config)
        return model, tokenizer
    
    else:
        return None, None
