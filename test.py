import json
import random

# 读取 JSON 数据
with open("dataset/output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 设置随机种子，保证可复现
random.seed(42)

# 随机打乱数据
random.shuffle(data)

# 计算划分索引
split_index = int(0.8 * len(data))  # 80% 训练集，20% 测试集

# 划分数据
train_data = data[:split_index]
test_data = data[split_index:]

# 保存数据
with open("train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")