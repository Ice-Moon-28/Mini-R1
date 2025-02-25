from reward.count_down_reward import equation_reward_func, format_reward_func
from reward.guess_word_reward import guess_word_reward_func, format_reward_func as format_reward_func_guess_word, length_reward_func


def get_reward(
        name='',
    ):
    if name == 'countdown':
        return [format_reward_func, equation_reward_func]
    elif name == 'guessing':
        return [format_reward_func_guess_word, guess_word_reward_func]
    else:
        return None