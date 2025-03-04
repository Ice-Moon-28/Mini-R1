from dataset.countdown_tasks_3to4 import get_countdown_collate_fn, get_countdown_dataset
from dataset.guess_word import get_guess_word_dataset, guess_word_collate_fn

def get_dataset(
        name='',
        tokenizer=None
    ):
    if name == 'countdown':
        return get_countdown_dataset(tokenizer=tokenizer)
    elif name == 'guessing':
        return get_guess_word_dataset(tokenizer=tokenizer)
    else:
        return None

def get_collect_fn(
        name="",
        tokenizer=None,
    ):
    if name == 'countdown':
        return lambda x: get_countdown_collate_fn(batch=x, tokenizer=tokenizer)
    elif name == 'guessing':
        return lambda x: guess_word_collate_fn(batch=x, tokenizer=tokenizer)
    else:
        return None

def get_kwargs_from_batch(
        name="",
):
    if name == 'countdown':
        return lambda x: {"nums": x['nums']}
    elif name == 'guessing':
        return lambda x: {}
    else:
        return lambda x: {}