import torch
from torch.onnx import export
from transformers import AutoTokenizer, AutoModelForCausalLM

# 定义为全局变量，用于统计 tokens
prompt_tokens = 0
completion_tokens = 0


class HuggingFaceModel:
    def __init__(self, model_path, device="cuda", torch_dtype="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
        self.device = device

    def __call__(self, prompt, max_new_tokens=32768, enable_thinking=False):
        global prompt_tokens, completion_tokens

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Count prompt tokens
        prompt_token_count = model_inputs["input_ids"].shape[-1]
        prompt_tokens += prompt_token_count

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Count completion tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        completion_token_count = len(output_ids)
        completion_tokens += completion_token_count

        # 解析 thinking 内容和最终输出内容
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # 查找 Thinking 分隔符 ID
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content

def get_usage(backend="gpt-4"):
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
