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
    normalized_length = (min((length - min_length) / (max_length - min_length), 1))

    # Use cosine to create a smooth reward curve
    # cos starts at -1 when x=0, goes to 1 when x=π
    # We transform this to go from 2.0 to 4.0
    reward = (normalized_length * base_reward)

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
                    think_process = re.search(f"<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>", completion)
                    if (len(think_process.groups()) == 1) and not has_repetition(think_process[0], min_len=30, max_repeats=3):
                        rewards.append(
                            1.0 + length_reward_enhancement(
                                completion=think_process[0],
                                base_reward=1.5,
                                min_length=0,
                                max_length=900,
                            )
                        )
                    else:
                        rewards.append(1.0)
                else:
                    rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    with open('countdown_chinese_format', "a+", encoding="utf-8") as f:
        for i in range(len(completions)):
            f.write("-" * 60 + "\n")
            comp = completions[i]
            answer = target[i]
            f.write(f"🤖 Output: {comp}\n")
            f.write(f"🎯 Target: {answer}\n")  

            f.write(f"🏆 Reward1: {rewards[i]}\n")
            f.write("-" * 60 + "\n")

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

            # think_process = re.search(f"<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>", completion)


            # # if len(completion) > 400:
            #     # Check for consecutive repeated substrings or "robot-like" responses
            # if (len(think_process.groups()) == 1) and not has_repetition(think_process[0]):

                
            #     rewards.append(
            #         2.0 + length_reward_enhancement(
            #             completion=think_process[0],
            #             base_reward=1,
            #             min_length=0,
            #             max_length=800,
            #         )
            #     )
            # else:
            rewards.append(2.0)
            # else:
            #     rewards.append(2.0)


            with open('countdown_chinese_right_accuracy', "a+", encoding="utf-8") as f:
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

    with open('countdown_chinese_accuracy', "a+", encoding="utf-8") as f:
        for i in range(len(completions)):
            f.write("-" * 60 + "\n")
            f.write(f"🤖 Output: {completion}\n")
            f.write(f"🎯 Target: {gt}\n")  
            f.write("-" * 60 + "\n")

    return rewards
if __name__ == "__main__":
    # 正确示例 1
    correct_sample_1 = """我们需要使用数字 19、36、55 和 7 各一次，并使用基本的算术运算，使其等于 65。一种可能的组合是 55 + 36 - 19 + 7... </think>
    <answer> 55 + 36 - 7 - 19 </answer>"""
    
    # 正确示例 2
    correct_sample_2 = """ ... </think><answer> 55 + 36 - 7 - 19 </answer>"""

    # 正确示例 3
    wrong_sample_3 = """ ... </think><think></think><answer> 55 + 36 - 7 - 19 </answer>"""

    # 正确示例 4（答案部分重复）
    wrong_sample_4 = """ ... </think><answer> 55 + 36 - 7 - 19 </answer><answer> 55 + 36 - 7 - 19 </answer>"""

    # 正确示例 5（答案部分包含等式）
    correct_sample_5 = """ ... </think><think></think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""

    # 正确示例 6（包含详细的思考过程）
    correct_sample_6 = """我需要使用基本的算术运算将数字 77、33、78 和 86 组合在一起，使其等于 52。一种可能的方法是使用减法和加法。我可以用 77 减去 33 得到 44，然后加上 78 得到 122。然而，这个数值太大了。因此，我可以用 86 减去 78 得到 8，然后再加上 44 得到 52。我需要创建一个使用给定数字并等于 98 的等式。我会从最大的数字 59 开始，看看是否能找到一种方法将其用于等式。然后，我会继续考虑其他数字。首先，我会先将最小的两个数字 56 和 83 相加，然后从最大的数字 90 中减去计算结果。</think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""

    # 正确示例 7（带有冗长重复的思考过程）
    correct_sample_7 = """ 我需要找到一个使用数字 19、36、55 和 7 各一次，并使用基本算术运算，使其等于 65 的等式。一种可能的组合是 55 + 36 - 19。我需要找到一个使用数字 19、36、55 和 7 各一次，并使用基本算术运算，使其等于 65 的等式。一种可能的组合是 55 + 36 - 19。我需要找到一个使用数字 19、36、55 和 7 各一次，并使用基本算术运算，使其等于 65 的等式。一种可能的组合是 55 + 36 - 19。我需要找到一个使用数字 19、36、55 和 7 各一次，并使用基本算术运算，使其等于 65 的等式。一种可能的组合是 55 + 36 - 19。</think><answer> 55 + 36 - 7 - 19 </answer>"""

    # 正确示例 8（带有乱码字符）
    correct_sample_8 = """ 我需要 t12121212cvbshaunsjckxnsajcnacjnsaocnasokjcnbasobsjfbsjkdbsajcbasdijbcqo 组合数字 77、33、78 和 86 并使用基本算术运算使其等于 52。12121291212 9121 我需要组合数字 77、33、78 和 86 以得到 52。一种可能的方法是使用减法和加法。我可以用 77 减去 33 得到 44，然后加上 78 得到 122。然而，这个数值太大了。因此，我可以用 86 减去 78 得到 8，然后再加上 44 得到 52。我需要创建一个使用给定数字并等于 98 的等式。我会从最大的数字 59 开始，看看是否能找到一种方法将其用于等式。然后，我会继续考虑其他数字。首先，我会先将最小的两个数字 56 和 83 相加，然后从最大的数字 90 中减去计算结果。</think><answer> 55 + 36 - 7 - 19 = 65 </answer>"""

    # 错误示例 5（计算错误）
    wrong_sample_5 = """ ... </think><think></think><answer> 55 + 36 - 7 + 19 = 65 </answer>"""
    
    # 错误格式示例 1（无 `<think>` 和 `<answer>` 标记）
    wrong_format = """用户：使用数字 [19, 36, 55, 7] 创建一个等于 65 的等式。"""

    # 错误格式示例 2（错误的 `think` 结构）
    wrong_format_2 = """要找到使用数字 95、78、6、88 并等于 79 的等式，我先将 88 和 95 相加：
    95 + 88 = 183
    现在，我们从 183 中减去 104 得到 79：
    183 - 104 = 79
    <think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""

    # 错误结果示例（错误答案）
    wrong_result = """ ... </think><answer> 55 + 36 - 7 - 18 </answer>"""

    # 测试格式奖励函数
    test_rewards = format_reward_func(
        completions=[
            correct_sample_1,
            correct_sample_2,
            wrong_format,
            wrong_format_2,
            wrong_result,
            wrong_sample_3,
            wrong_sample_4,
            correct_sample_7,
            correct_sample_8,
        ],
        target=["65"] * 9,
        nums=[[19, 36, 55, 7]] * 9,
    )
    print(test_rewards)
    assert test_rewards == [1.267, 1.06, 0.0, 0.0, 1.06, 0.0, 0.0, 1.0, 2.173], "奖励函数工作不正确"

    # 测试数学计算奖励函数
    test_rewards = equation_reward_func(
        completions=[correct_sample_1, correct_sample_2, correct_sample_6, correct_sample_7, correct_sample_8, wrong_format, wrong_format_2, wrong_result, correct_sample_5, wrong_sample_5],
        target=["65"] * 10,
        nums=[[19, 36, 55, 7]] * 10
    )
    print(test_rewards)
    assert test_rewards == [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0], "奖励函数工作不正确"