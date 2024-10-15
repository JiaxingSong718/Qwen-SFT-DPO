import torch

# 训练数据预处理方法
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