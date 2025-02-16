import json
import re

import torch
from datasets import Dataset
import pandas as pd
import os
from transformers import AutoTokenizer

def classify_type(text):
    """检查文本是否包含 '1. *** 2. *** 3. *** 4. ***' 形式的内容"""
    if pd.isna(text):
        return "question"
    # 正则匹配严格的 '1. xxx 2. xxx 3. xxx 4. xxx' 格式
    pattern = r"1\..+?2\..+?3\..+?4\..+?"
    return "association" if re.search(pattern, text) else "question"
class GuessWordDataset(Dataset):
    def __init__(self, file_path):
        """
        初始化 GuessWordDataset 类，继承自 datasets.Dataset
        :param word_list: 一个包含单词字典的列表，每个字典至少包含 'term' 和 'definition' 键
        """
        self.df = pd.read_excel(file_path)

        self.datas = []
        for row in self.df.itertuples(index=False, name=None):
            
            index, class_word, source, answer, question = row[:5]
            clean_text = question.replace("\xa0", " ")

            item = {
                'index': index,
                'class': class_word,
                'source': source,
                'answer': answer,
                'question': clean_text,
                'label': classify_type(clean_text),
            }

            self.datas.append(
                item
            )

            print(f"Index: {index}, Term: {item}")


        # 使用 from_dict 创建 Hugging Face 数据集
        dataset = Dataset.from_dict({key: [d[key] for d in self.datas] for key in self.datas[0]})
        super().__init__(dataset.data, dataset.info)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的单词项
        :param idx: 索引
        :return: 包含 'term' 和 'definition' 的字典
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.datas[idx]

    def __len__(self):
        """
        获取数据集的长度
        :return: 数据集中的单词总数
        """
        return len(self.datas)

    def to_json(self, output_path='output.json'):
        """
        将数据集保存为格式化的 JSON 文件
        :param output_path: 输出 JSON 文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.datas, f, indent=4, ensure_ascii=False)
        print(f"JSON 文件已保存至: {output_path}")

    @classmethod
    def read_from_json(self, file_path="output.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            dataset = Dataset.from_dict({
                "index": [item["index"] for item in data],
                "class": [item["class"] for item in data],
                "source": [item["source"] for item in data],
                "answer": [item["answer"] for item in data],
                "question": [item["question"] for item in data],
                "label": [item["label"] for item in data],
            })

            return dataset




def get_guess_word_dataset(tokenizer):
    def generate_association_prompt(question, answer, description):
        description = re.sub(r"[【】\s]", "", description)

        assert len(description) <= 3

        initial_letter = description[0]

        word_length = description[1:]

        # question_prefix = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "你是一个善于推理和联想的中文语言助手。"
        #             "在回答之前会在脑中先思考推理过程，然后再为用户提供答案。"
        #         )
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"请根据以下提示词语，推测出对应的词语或诗句：\n{question}\n\n"
        #             "提示信息：\n"
        #             f"- 目标答案的字数为：{word_length} 字\n\n"
        #             f"- 目标答案的首字拼音声母为：'{initial_letter}'\n"
        #             "每个提示词前面的数字表示该提示的编号。"
        #             "请综合所有提示内容进行推理。\n\n"
        #             "请在 <think> </think> 标签内详细描述你的推理过程，并在 <answer> </answer> 标签内给出最终答案。"
        #         )
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "让我一步步推理。\n<think>"
        #     }
        # ]

        question_prefix = [
            {
                "role": "system",
                "content": (
                    "你是一个乐于助人的助手。在回答之前会在脑中先思考推理过程，然后再为用户提供答案。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"请根据以下描述，推测出对应的词语或者是诗句: \n{question}\n\n"
                    # "提示信息：\n"
                    # f"- 目标答案的字数为：{word_length} 字\n\n"
                    # f"- 目标答案的首字拼音声母为：'{initial_letter}'\n"
                    # "每个提示词前面的数字表示该提示的编号。"
                    # "请综合所有提示内容进行推理。\n\n"
                    "请在<think> </think>标签内展示推理过程，并在<answer> </answer>标签内给出答案。"
                )
            },
            {
                "role": "assistant",
                "content": "让我一步步推理。\n<think>"
            }
        ]

        
        return {
            "prompt": tokenizer.apply_chat_template(question_prefix, tokenize=False, continue_final_message=True),
            "target": answer.split('/'),
            "label": 'question',
        }

    def generate_question_prompt(question, answer, description):
        description = re.sub(r"[【】\s]", "", description)

        initial_letter = description[0]

        word_length = description[1:]

        assert len(description) <= 3

        question_prefix = [
            {
                "role": "system",
                "content": (
                    "你是一个乐于助人的助手。在回答之前会在脑中先思考推理过程，然后再为用户提供答案。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"以下提示词: \n{question}\n\n"
                    # "提示信息：\n"
                    # f"- 目标答案的字数为：{word_length} 字\n\n"
                    # f"- 目标答案的首字拼音声母为：'{initial_letter}'\n"
                    # "每个提示词前面的数字表示该提示的编号。"
                    # "请综合所有提示内容进行推理。\n\n"
                    "请推测出与这些提示相关的词语或者是诗句。"
                    "请在<think> </think>标签内展示推理过程，并在<answer> </answer>标签内给出答案。"
                )
            },
            {
                "role": "assistant",
                "content": "让我一步步推理。\n<think>"
            }
        ]

        # question_prefix = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "你是一个善于推理和联想的中文语言助手。"
        #             "在回答之前会在脑中先思考推理过程，然后再为用户提供答案。"
        #         )
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"以下是一个描述性提示，请根据描述推测出对应的词语或诗句：\n\n"
        #             f"{question}\n\n"
        #             "提示信息：\n"
        #             f"- 答案的字数为：{word_length} 字\n\n"
        #             f"- 答案的首个字的拼音声母为：'{initial_letter}'\n"
        #             "请在 <think> </think> 标签内展示推理过程，并在 <answer> </answer> 标签内给出最终答案。"
        #         )
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "让我一步步推理。\n<think>"
        #     }
        # ]

        return {
            "prompt": tokenizer.apply_chat_template(question_prefix, tokenize=False, continue_final_message=True),
            "target": answer.split('/'),
            'label': 'association',
        }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    relative_path = os.path.join(script_dir, "output.json")

    dataset = GuessWordDataset.read_from_json(relative_path)

    original_columns = dataset.column_names

    dataset = dataset.map(
        lambda x: generate_question_prompt(question=x['question'], answer=x['answer'], description=x['class']) if x["label"] == "question" else generate_association_prompt(question=x['question'], answer=x['answer'], description=x['class']),
        remove_columns=original_columns,
    )

    return dataset



def guess_word_collate_fn(batch, tokenizer):
  
    # 1️⃣ 提取字段
    prompts = [item['prompt'] for item in batch]
    targets = [item['target'] for item in batch]
    labels = [item['label'] for item in batch]

    # 2️⃣ 将 prompt 转换为模型输入格式（padding）
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")


    # 4️⃣ 将 label 转换为张量（字符串无法直接转为张量，因此用数值编码）
    label_map = {"question": 0, "association": 1}
    label_ids = torch.tensor([label_map.get(label, -1) for label in labels])

    # 5️⃣ 返回一个包含所有信息的批次字典
    return {
        "prompts": {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        },
        "target": targets,
        "labels": label_ids,
        "input": prompts,
    }


if __name__ == '__main__':
    # dataset = GuessWordDataset('guess_word.xlsx')

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = get_guess_word_dataset(tokenizer=tokenizer)