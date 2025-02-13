from dataset.countdown_tasks_3to4 import get_countdown_dataset

def get_dataset(
        name='',
        tokenizer=None
    ):
    if name == 'countdown':
        return get_countdown_dataset(tokenizer=tokenizer)
    elif name == 'guessing':
        return None
    else:
        return None