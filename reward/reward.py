def has_repetition(text, min_len=50, max_repeats=3):
    """
    Check if the text contains consecutive repeating substrings or phrases.
    
    Args:
        text (str): The string to check.
        min_len (int): Minimum length of the substring to check for repetition.
        max_repeats (int): Maximum number of repeats allowed for any substring.
    
    Returns:
        bool: True if there is repetition, False otherwise.
    """
    # Check for repeating substrings of length >= min_len
    for length in range(min_len, len(text) // 2 + 1):
        for i in range(len(text) - length):
            substring = text[i:i + length]
            repeats = text.count(substring)
            if repeats > max_repeats:
                return True
    return False

def length_reward_enhancement(completion, base_reward=2, min_length=0, max_length=1000):
    """
    Calculate a reward based on the length of the completion using a cosine function.
    
    Parameters:
    - completion: The text to evaluate
    - min_length: Minimum desired length (default 200)
    - max_length: Maximum desired length (default 800)
    
    Returns:
    - A reward value between 2.0 and 4.0
    """
    length = len(completion)
    
    # If length is outside the desired range, use a base reward
    if length < min_length:
        return 0
    
    # Normalize the length to a 0 to π range
    normalized_length = (min((length - min_length) / (max_length - min_length), 1))

    # Use cosine to create a smooth reward curve
    # cos starts at -1 when x=0, goes to 1 when x=π
    # We transform this to go from 2.0 to 4.0
    reward = (normalized_length * base_reward)

    return reward
