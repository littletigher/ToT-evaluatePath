import os
import openai
import backoff 

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def huggingfaceModel(prompt, model="D:\Github\module\Qwen3-4B", device="cuda", max_new_tokens=32768, n=1, stop=None,
                enable_thinking=False):
    """
        调用本地 HuggingFace chat 模型（如 Qwen）进行推理。

        :param prompt: 输入提示文本
        :param model_name: 本地模型路径或 HuggingFace 模型名
        :param max_new_tokens: 最大生成 token 数量
        :param enable_thinking: 是否启用思维链模式（Thinking Mode）
        :return: (thinking_content, content) 两个部分的内容
        """
    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model)
    hfModel = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        device_map="auto"
    )

    # 构造 messages 并应用 chat template
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(hfModel.device)

    # 进行推理
    with torch.no_grad():
        generated_ids = hfModel.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 解析 thinking 内容和最终输出内容
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # 查找 Thinking 分隔符 ID
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content



def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.00250 + prompt_tokens / 1000 * 0.01
    elif backend == "qwen3-4B":
        cost = "LOCAL EXECUTION"
        # cost = completion_tokens / 1000 * 0.001 + prompt_tokens / 1000 * 0.0005
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
