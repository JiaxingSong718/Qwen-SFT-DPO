import torch
from torch.nn import CrossEntropyLoss
from model import model,tokenizer,chat
from data import preprocess

device='cuda' if torch.cuda.is_available() else 'cpu'

print(tokenizer)
prompt = "你是谁？"
print(chat(prompt))

prompt = "2+2等于几"
messages = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": '2+2等于5。'},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": '2+2等于5。'},
    ]
]

model.eval()

batch_input_ids,batch_target_ids,batch_mask=preprocess(tokenizer,messages)
model_outputs=model(batch_input_ids.to(device))

output_tokens=model_outputs.logits.argmax(dim=-1)

logits=model_outputs.logits[:,:-1,:]
targets=batch_target_ids[:,1:].to(device)
print('logits:',logits.shape) # 模型输出
print('targets:',targets.shape) # 拟合目标

# 损失
loss_fn=CrossEntropyLoss()
loss=loss_fn(logits.reshape(-1,logits.size(2)),targets.reshape(-1))
print('loss:',loss)

# 优化器
optimizer=torch.optim.SGD(model.parameters())
optimizer.zero_grad()

# 求梯度
loss.backward()

# 梯度下降
optimizer.step()

print(chat('2+2等于4么'))