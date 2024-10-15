import torch

dpo_train_data=[
    {'prompt':'你是谁?','chosen':'通义千问','reject':'我是阿里云开发的超大规模语言模型，我叫通义千问。'},
    {'prompt':'你是谁发明的?','chosen':'张三','reject':'阿里巴巴'},
]

# 偏好数据集 -> 模型输入
def dpo_to_messages(dpo_pairs):
    chosen_messages=[]
    reject_messages=[]
    for pair in dpo_pairs:
        chosen_messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pair['prompt']},
                {"role": "assistant", "content": pair['chosen']},
            ]
        )
        reject_messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pair['prompt']},
                {"role": "assistant", "content": pair['reject']},
            ]
        )
    return chosen_messages,reject_messages

# 训练数据预处理
def preprocess(tokenizer,batch_messages):
    input_list=[]
    target_list=[]
    
    im_start=tokenizer('<|im_start|>').input_ids
    im_end=tokenizer('<|im_end|>').input_ids
    newline=tokenizer('\n').input_ids
    pad=tokenizer('<|endoftext|>').input_ids
    ignore=[-100]
    
    for group in batch_messages:
        input_ids=[]
        target_ids=[]
        for msg in group:
            role=tokenizer(msg['role']).input_ids
            content=tokenizer(msg['content']).input_ids
            if msg['role'] in ['system','user']:
                ignore_parts=role+newline+content
                input_ids+=im_start+ignore_parts+im_end+newline
                target_ids+=im_start+ignore*len(ignore_parts)+im_end+newline
            else:
                ignore_parts=role+newline
                input_ids+=im_start+ignore_parts+content+im_end+newline
                target_ids+=im_start+ignore*len(ignore_parts)+content+im_end+newline
        input_list.append(input_ids)
        target_list.append(target_ids)
    
    # padding
    max_len=max([len(ids) for ids in input_list])
    for input_ids,target_ids in zip(input_list,target_list):
        input_ids+=pad*(max_len-len(input_ids))
        target_ids+=ignore*(max_len-len(target_ids))
    batch_input_ids=torch.tensor(input_list,dtype=torch.long)
    batch_target_ids=torch.tensor(target_list,dtype=torch.long)
    batch_mask=batch_input_ids.ne(pad[0]).type(torch.long)
    return batch_input_ids,batch_target_ids,batch_mask