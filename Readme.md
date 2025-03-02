## 安装环境

### 目前的规划

| 序号 | 任务描述 | 评估方式 | 是否完成 |
|----|------------------------------------------------------|---------------------------------|----------|
| 1  | 尝试复现 [Mini DeepSeek R1](https://www.philschmid.de/mini-deepseek-r1) 中的 `aha moment` | 原有数据集的 valid 上的正确率 |    No     |
| 2  | 对 **猜词游戏数据集** 进行 GRPO 强化训练，期望其能够实现更高的正确几率 | 计划利用原有数据集上的 valid 正确率来进行评估 |    Yes      |


### 前期评估结果

中文数据集
https://huggingface.co/datasets/icemoon28/guess_word_dataset
#### 3B Instruct Model 效果
1. Reward Format: 2687/3070
2. Reward Accuracy: 744/3070
#### 3B Base Model 效果
1. Reward Format: 582/3070
2. Reward Accuracy: 89/3070
#### 7B Instruct Model 效果
1. Reward Format: 2903/3070
2. Reward Accuracy: 1229/3070
#### 7B Base Model 效果
1. Reward Format: 3023/3070
2. Reward Accuracy: 881/3070

数学计算数据集
#### 3B Instruct Model 效果
1. Reward Format: 
2. Reward Accuracy:
#### 3B Base Model 效果
1. Reward Format: 
2. Reward Accuracy: 
#### 7B Instruct Model 效果
1. Reward Format: 
2. Reward Accuracy: 
#### 7B Base Model 效果
1. Reward Format:
2. Reward Accuracy: 


### 中文数据集训练结果

#### Base 3B model 训练后
  1. Reward Format: 3051/3070
  2. Reward Accuracy: 1187/3070
#### 结论
学习出了 Pattern，但是没有学习出来 自我反思的能力
```
回答倾向于在 answer 中总结推理过程
🤖 Output: system
你是一个乐于助人的助手。在回答之前会在脑中先思考推理过程，然后再为用户提供答案。
user
以下提示词: 
这种语言是欧洲某国的官方语言之一，也是其邻国的法定少数民族语言，是黏着语，不同于大多数欧洲语言的是，该语言不属于印欧语系，而是属于乌拉尔语系。

请推测出与这些提示相关的词语或者是诗句。请在<think> </think>标签内展示推理过程，并在<answer> </answer>标签内给出答案。
assistant
让我一步步推理。
<think> 根据提示，这种语言是欧洲某国的官方语言之一，也是其邻国的法定少数民族语言，是黏着语，不属于印欧语系，而是属于乌拉尔语系。这些特征指向了芬兰语。 </think>

<answer> 根据推理过程，与这些提示相关的词语是芬兰语。芬兰语是芬兰的官方语言，也是其邻国瑞典的法定少数民族语言。芬兰语是一种黏着语，属于乌拉尔语系，而不是印欧语系。</answer>
🎯 Target: ['芬兰语']
🏆 Reward0: 1.0
🏆 Reward1: 1.5
------------------------------------------------------------
------------------------------------------------------------
🤖 Output: system
你是一个乐于助人的助手。在回答之前会在脑中先思考推理过程，然后再为用户提供答案。
user
以下提示词: 
这座城市是某个欧洲国家的首都，始建于公元1263年，1981年与天津市结为友好城市，这座城市发生的一个历史事件是近代一次世界大战的导火索。

请推测出与这些提示相关的词语或者是诗句。请在<think> </think>标签内展示推理过程，并在<answer> </answer>标签内给出答案。
assistant
让我一步步推理。
<think> 根据提示，这座城市是某个欧洲国家的首都，始建于公元1263年，1981年与天津市结为友好城市，这座城市发生的一个历史事件是近代一次世界大战的导火索。因此，这座城市可能是巴黎，因为它是法国的首都，始建于12th世纪，1981年与天津市结为友好城市，1914年的萨拉热窝事件是第一次世界大战的导火索。 </think>

<answer> 根据推理过程，与这些提示相关的词语或者是诗句是“巴黎”、“萨拉热窝事件”。</answer>
🎯 Target: ['萨拉热窝']
🏆 Reward0: 1.0
🏆 Reward1: 1.5
```


### 具体的记录文档
https://k3x3xq5vv8.feishu.cn/docx/N9KTdzWtkoALmmx4da6cyVFHnSg?from=from_copylink