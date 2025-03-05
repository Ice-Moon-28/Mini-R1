import math
import re
 

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
    normalized_length = (min((length - min_length) / (max_length - min_length), 1)) * math.pi

    # Use cosine to create a smooth reward curve
    # cos starts at -1 when x=0, goes to 1 when x=π
    # We transform this to go from 2.0 to 4.0
    reward = base_reward + math.cos(normalized_length - math.pi) * base_reward

    return reward

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


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []
 
    for completion in completions:
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion        
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>[\s]*<answer>([\s\S]*?)<\/answer>$"
    
            match = re.search(regex, completion, re.DOTALL) 
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                answer_pattern = r"<answer>([\s\S]*?)<\/answer>"

                answer_matches = re.findall(answer_pattern, completion, re.DOTALL)

                if len(answer_matches) == 1:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards
 
def equation_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer
 
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers
    
    Returns:
        list[float]: Reward scores
    """

    nums = kwargs["nums"]

    assert nums != None

    rewards = []

    for completion, gt, numbers in zip(completions, target, nums):
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            rewards.append(0.0)
            continue
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()


        # 取左边
        if '=' in equation:
            equation = equation.split('=')[0].strip()

        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            rewards.append(0.0)
            continue
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
           rewards.append(0.0)
           continue
        
        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(gt)) < 1e-5:

            think_process = re.search(f"<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>", completion)


            # if len(completion) > 400:
                # Check for consecutive repeated substrings or "robot-like" responses
            if (len(think_process.groups()) == 1) and not has_repetition(think_process[0]):

                
                rewards.append(
                    2.0 + length_reward_enhancement(
                        completion=think_process[0],
                        base_reward=1,
                        min_length=0,
                        max_length=800,
                    )
                )
            else:
                rewards.append(2.0)
            # else:
            #     rewards.append(2.0)


            with open('countdown_accuracy', "a+", encoding="utf-8") as f:
                for i in range(len(completions)):
                    f.write("-" * 60 + "\n")
                    f.write(f"🤖 Output: {completion}\n")
                    f.write(f"🎯 Target: {gt}\n")  
                    f.write("-" * 60 + "\n")
        else:
            rewards.append(0.0)
      except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0) 

    return rewards

if __name__ == "__main__":
    correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19 + 7... </think>
    <answer> 55 + 36 - 7 - 19 </answer>"""
    
    correct_sample_2 = """ ... </think><answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_3 = """ ... </think><think></think><answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_4 = """ ... </think><answer> 55 + 36 - 7 - 19 </answer><answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_5 = """ ... </think><think></think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""

    correct_sample_6 = """I need to combine the numbers 77, 33, 78, and 86 with basic arithmetic to get the answer 52. One possible way to do this is to use subtraction and addition. I can subtract 33 from 77 to get 44, and then add 78 to get 122. However, this is too high. So, I can subtract 78 from 86 to get 8, and then add 44 to get 52.  I need to create an equation using the given numbers that equals 98. I'll start by looking at the largest number, 59, and see if I can find a way to use it in the equation. Then I'll move on to the other numbers. First, I'll start by adding the two smallest numbers, which are 56 and 83. Then, I'll subtract the result from the largest number, which is 90.</think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""

    correct_sample_7 = """ to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19  to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19  to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19  to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19  to find an equation using the numbers 19, 36, 55, and 7 exactly once, with basic arithmetic operations, that equals 65. One possible combination is 55 + 36 - 19 </think><answer> 55 + 36 - 7 - 19 </answer>"""

    correct_sample_8 =  """ I need t12121212cvbshaunsjckxnsajcnacjnsaocnasokjcnbasobsjfbsjkdbsajcbasdijbcqo combine the numbers 77, 33, 78, and 86 with b12121291212 9121 I need to combine the numbers 77, 33, 78, and 86 with basic arithmetic to get the answer 52. One possible way to do this is to use subtraction and addition. I can subtract 33 from 77 to get 44, and then add 78 to get 122. However, this is too high. So, I can subtract 78 from 86 to get 8, and then add 44 to get 52.  I need to create an equation using the given numbers that equals 98. I'll start by looking at the largest number, 59, and see if I can find a way to use it in the equation. Then I'll move on to the other numbers. First, I'll start by adding the two smallest numbers, which are 56 and 83. Then, I'll subtract the result from the largest number, which is 90.</think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""


    wrong_sample_5 = """ ... </think><think></think><answer> 55 + 36 - 7 + 19 = 65 </answer>"""
    
    wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
    
    wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
    95 + 88 = 183                                                                                                              
    Now, let's subtract 104 from 183 to get 79:
    183 - 104 = 79
    <think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""
    
    wrong_result = """ ... </think><answer> 55 + 36 - 7 - 18 </answer>"""

    test_rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result, correct_sample_3, correct_sample_4], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    print(test_rewards)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], "Reward function is not working"
    test_rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, correct_sample_6, correct_sample_7, correct_sample_8, wrong_format, wrong_format_2, wrong_result, correct_sample_5, wrong_sample_5], target=["65"] * 10, nums=[[19, 36, 55, 7]] * 10)
    print(test_rewards)
    assert test_rewards == [2.536697580176151, 2.006165332533744, 5.809654104932039, 2.0, 6.0, 0.0, 0.0, 0.0, 2.006165332533744, 0.0], "Reward function is not working"

