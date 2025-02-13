import json
import re
from datasets import Dataset
import pandas as pd

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



if __name__ == '__main__':
    dataset = GuessWordDataset('guess_word.xlsx')

    dataset.to_json()

    import pdb; pdb.set_trace()