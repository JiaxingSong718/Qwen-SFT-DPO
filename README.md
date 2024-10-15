# Qwen2-SFT&DPO

## 1.大模型训练过程

### 1.1 预训练(Pre-Train)

使用无标注的海量语料训练模型的接龙能力，读尽天下的文章。

```
训练语料：
<|beginoftext|>浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。<|endoftext|>
```

### 1.2 监督式微调(Supervised Fine-Tuning, SFT)

使用少量有标注的高质量语料，锻炼模型的问答能力。

```
训练语料：
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请介绍一下浙江大学<|im_end|>\n<|im_start|>assistant\n浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。<|im_end|>

推理阶段：
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请介绍一下浙江大学<|im_end|>\n<|im_start|>assistant\n
大模型返回：
浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。<|im_end|>\n
```

### 1.3 直接偏好优化(Direct Preference Optimization, DPO)

**RLHF与DPO的关系**

+ DPO（Direct Preference Optimization）和RLHF（Reinforcement Learning from Human Feedback）都是用于训练和优化人工智能模型的方法，特别是在大型语言模型的训练中
+ DPO和RLHF都旨在通过人类的反馈来优化模型的表现，它们都试图让模型学习到更符合人类偏好的行为或输出
+ RLHF通常涉及三个阶段：全监督微调（Supervised Fine-Tuning）、奖励模型（Reward Model）的训练，以及强化学习（Reinforcement Learning）的微调
+ DPO是一种直接优化模型偏好的方法，不需要显式地定义奖励函数，而是通过比较不同模型输出的结果，选择更符合人类偏好的结果作为训练目标，主要是通过直接最小化或最大化目标函数来实现优化，利用偏好直接指导优化过程，而不依赖于强化学习框架

**DPO的训练目标**

$$r(x,y)$$为奖励函数，$$\pi_{ref}(y|x)$$为基准模型，$$\pi(y|x)$$训练模型。

训练目标为
$$
\max_\pi\mathbb{E}_{x\sim D,y\sim\pi}\left[r(x,y)\right]-\beta\mathbb{D}_{KL}[\pi(y|x)||\pi_{ref}(y|x)]\\
$$

+ 左半部分得到尽可能高的奖励
+ 右半部分防止模型偏离奖励模型准确分布太远，同时保持生成的多样性

最终化简的DPO Loss为
$$
-ln\sigma(\beta ln\frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)}-\beta ln\frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)})\\\sigma(x)=\frac1{1+exp(-x)}
$$


```
训练语料：
1）chosen好的回答：
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n请介绍一下浙江大学<|im_end|>\n
<|im_start|>assistant\n浙江大学是一所历史悠久、声誉卓著的高等学府，坐落于中国历史文化名城、风景旅游胜地杭州。<|im_end|>\n

2）rejected坏的回答：
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
<|im_start|>user\n请介绍一下浙江大学<|im_end|>\n
<|im_start|>assistant\n浙江大学是一所野鸡大学。<|im_end|>\n

【偏好对齐】通过DPO算法，可以让大模型回答的时候，倾向于返回chosen这种回答，尽量不用rejected这种口吻。
```

## 2.Qwen2-SFT

`Qwen2-0.5B-Instruct`的SFT，在./SFT中

## 3.Qwen2-DPO

`Qwen2-0.5B-Instruct`的DPO，在./DPO中
