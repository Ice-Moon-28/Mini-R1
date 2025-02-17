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
            regex = r"<think>([\s\S]*?)<\/think>[\s]*<answer>([\s\S]*?)<\/answer>"
    
            match = re.search(regex, completion, re.DOTALL) 
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)

    print(rewards)

    return rewards

def guess_word_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on:
    1. Correctness of the guessed word with <answer> tags
    
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):
        try:
            completion = "<think>" + completion   

            answer_match = re.findall(r"<answer>([\s\S]*?)<\/answer>", completion, re.DOTALL)

            if answer_match is None:
                rewards.append(0.0)
                continue

            answers = [ans.strip() for ans in answer_match]

            matched = False

            if isinstance(gt, str):
                # gt 为字符串的处理逻辑
                for answer in answers:
                    if gt in answer:
                        rewards.append(1.0)
                        matched = True
                        break

            elif isinstance(gt, list):
                # gt 为列表时，任意一个元素匹配即得分
                for answer in answers:
                    match_found = any(target in answer for target in gt)
                    if match_found:
                        rewards.append(1.0)
                        matched = True
                        break

            if not matched:
                rewards.append(0.0)

        except Exception:
            rewards.append(0.0)

    return rewards




if __name__ == "__main__":
    correct_completion = """
    手游兴盛，叠词，通常用于描述事物蓬勃发展，答案应是‘蒸蒸日上’。</think>
    <answer>蒸蒸日上</answer>
    """

    correct_completion2 = """
   描述受人敬仰的成语，含有地名和星宿名。泰山和北斗均为具有代表性的元素，答案是‘泰山北斗’。</think>
    <answer>泰山北斗</answer>
    """


    correct_completion3 = """
    首先，我们来分析每个提示词的含义和可能的关联领域：</think>
    1. 黑色 - 这个词通常与颜色相关，常见于描述物品的颜色属性。
    2. 金 - 这个词也与颜色有关，指的是金色，常用于形容贵金属。
    3. 鲁滨逊 - 这个词是人名，但更有可能是指英国小说家丹尼尔·笛福的著作《鲁滨逊漂流记》中的主人公。
    4. 金曜日 - 这个词是日语，意思是“星期五”，是日本对星期五的日语表达。
    根据以上分析，我们可以推测出与这些提示相关的词语：
    - 黑色 - 可能与夜、夜晚、黑色的物体等有关。
    - 金 - 可能与黄金、金币、金色等有关。
    - 鲁滨逊 - 可能与冒险、航海、荒岛生存等有关。
    - 金曜日 - 可能与星期五、周五等日期有关。
    <answer>
    黑色：夜、夜晚、黑色的物体
    金：黄金、金币、金色
    鲁滨逊：冒险、航海、荒岛生存
    金曜日：星期五、周五
    </answer>
    """

    missing_answer = """
    手游兴盛，叠词，答案可能是‘蒸蒸日上’。</think>
    """

    wrong_answer = """
    手游兴盛，叠词，答案可能是‘风生水起’。</think>
    <answer>风生水起</answer>
    """

    test_rewards = guess_word_reward_func(
        completions=[correct_completion, correct_completion2, correct_completion3, correct_completion, correct_completion2, correct_completion3, missing_answer, wrong_answer],
        target=["蒸蒸日上", "泰山北斗", "星期五", ["蒸蒸日上", "泰山北斗"] , ["泰山北斗", "蒸蒸日上"], ["蒸蒸日上", "泰山北斗"], "蒸蒸日上", "蒸蒸日上"]
    )

    test_reward_format = format_reward_func(
        completions=[correct_completion, correct_completion2, correct_completion3, correct_completion, correct_completion2, correct_completion3, missing_answer, wrong_answer],
        target=["蒸蒸日上", "泰山北斗", "星期五", ["蒸蒸日上", "泰山北斗"] , ["泰山北斗", "蒸蒸日上"], ["蒸蒸日上", "泰山北斗"], "蒸蒸日上", "蒸蒸日上"]
    )

    print("Test Rewards:", test_rewards)
    assert test_rewards == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "Reward function is not working correctly."

    print("Test format Rewards", test_reward_format)
    assert test_reward_format == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], "Reward function is not working correctly."