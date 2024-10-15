import torch
from modelscope import AutoModelForCausalLM,AutoTokenizer

device='cuda' if torch.cuda.is_available() else 'cpu'

def create_qwen_model():
    model = AutoModelForCausalLM.from_pretrained(
        "qwen/Qwen2-0.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-0.5B-Instruct")
    return model,tokenizer


# 模型测试方法
def chat(prompt,tokenizer,model):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    #print(text)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response