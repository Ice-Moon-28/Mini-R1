import re
 
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
            rewards.append(2.0)
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

    wrong_sample_5 = """ ... </think><think></think><answer> 55 + 36 - 7 + 19 = 65 </answer>"""
    
    wrong_format = """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
    
    wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:                      
    95 + 88 = 183                                                                                                              
    Now, let's subtract 104 from 183 to get 79:
    183 - 104 = 79
    <think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""
    
    wrong_result = """ ... </think><answer> 55 + 36 - 7 - 18 </answer>"""
    
    
    test_rewards = format_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result, correct_sample_3, correct_sample_4], target=["65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 5)
    assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5], "Reward function is not working"
    test_rewards = equation_reward_func(completions=[correct_sample_1, correct_sample_2, wrong_format, wrong_format_2, wrong_result, correct_sample_5, wrong_sample_5], target=["65", "65", "65", "65", "65", "65", "65"], nums=[[19, 36, 55, 7]] * 7)
    assert test_rewards == [2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0], "Reward function is not working"