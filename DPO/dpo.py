import torch
from model import create_qwen_model,chat
from data import dpo_to_messages,dpo_train_data,preprocess

device='cuda' if torch.cuda.is_available() else 'cpu'

# DPO训练的模型
model_pi,tokenizer=create_qwen_model()
# DPO参照的模型
model_ref,_=create_qwen_model()

model_pi.train()
model_ref.train()
# 优化器，只训练pi模型
optimizer=torch.optim.SGD(model_pi.parameters(),lr=1e-3)
# DPO损失计算-辅助函数
def dpo_prob_calc(target_ids,pi_logits,ref_logits):
    pi_probs=torch.log_softmax(pi_logits,dim=-1)      # softmax概率+log对数
    ref_probs=torch.log_softmax(ref_logits,dim=-1)
    
    ignore_mask=target_ids!=-100 # ignore token掩码
    indexes=target_ids*ignore_mask # 将-100变成0，以便后面gather可以运行
    
    pi_probs_of_target=torch.gather(pi_probs,dim=-1,index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask # 取目标target token的概率，忽略-100 token
    ref_probs_of_target=torch.gather(ref_probs,dim=-1,index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask    
    
    pi_final_prob=pi_probs_of_target.sum(-1)/ignore_mask.sum(-1)     # 求每一个样本的token prob均值
    ref_final_prob=ref_probs_of_target.sum(-1)/ignore_mask.sum(-1)
    return pi_final_prob,ref_final_prob
    
# DPO损失函数 https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
def dpo_loss(params):
    ## 两个模型的chosen输出
    chosen_target_ids=params['chosen_target_ids'][:,1:]
    pi_chosen_logits=params['pi_chosen_logits'][:,:-1,:]
    ref_chosen_logits=params['ref_chosen_logits'][:,:-1,:]
    pi_chosen_prob,ref_chosen_prob=dpo_prob_calc(chosen_target_ids,pi_chosen_logits,ref_chosen_logits)
    
    ## 两个模型的reject输出
    reject_target_ids=params['reject_target_ids'][:,1:]
    pi_reject_logits=params['pi_reject_logits'][:,:-1,:]
    ref_reject_logits=params['ref_reject_logits'][:,:-1,:]
    pi_reject_prob,ref_reject_prob=dpo_prob_calc(reject_target_ids,pi_reject_logits,ref_reject_logits)
    
    # 计算DPO Loss
    pi_prob_diff=pi_chosen_prob-pi_reject_prob 
    ref_prob_diff=ref_chosen_prob-ref_reject_prob
    beta=0.1
    loss=-torch.nn.functional.logsigmoid(beta*(pi_prob_diff-ref_prob_diff))
    return loss.mean()

iterators=20

vocab=tokenizer.get_vocab()
for i in range(iterators):
    # 一批模拟数据
    chosen_messages,reject_messages=dpo_to_messages(dpo_train_data)
    # model输入和输出
    chosen_input_ids,chosen_target_ids,chosen_mask=preprocess(tokenizer,chosen_messages)
    reject_input_ids,reject_target_ids,reject_mask=preprocess(tokenizer,reject_messages)
    # model_pi预测
    pi_chosen_logits=model_pi(input_ids=chosen_input_ids.to(device),attention_mask=chosen_mask.to(device)).logits
    pi_reject_logits=model_pi(input_ids=reject_input_ids.to(device),attention_mask=reject_mask.to(device)).logits
    # model_ref预测
    ref_chosen_logits=model_ref(chosen_input_ids.to(device),chosen_mask.to(device)).logits
    ref_reject_logits=model_ref(reject_input_ids.to(device),reject_mask.to(device)).logits
    # 求DPO损失
    loss=dpo_loss({
        'chosen_target_ids':chosen_target_ids.to(device),
        'reject_target_ids':reject_target_ids.to(device),
        'pi_chosen_logits':pi_chosen_logits.to(device),
        'pi_reject_logits':pi_reject_logits.to(device),
        'ref_chosen_logits':ref_chosen_logits.to(device),
        'ref_reject_logits':ref_reject_logits.to(device),
    })
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_pi.eval()
print(chat('你是谁?',tokenizer,model_pi))

print(chat('你是谁发明的?',tokenizer,model_pi))

print(chat('讲讲transformer模型',tokenizer,model_pi))