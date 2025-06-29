import openai
import time
import random
# 获取当前脚本的父目录（即项目根目录）
import sys
import os
from tot.tasks import get_task
# 替换为你的 OpenAI API Key
openai.api_key = "sk-3N7pMMZwWkqu1HyZgEr8AoM0EM8l5mvNx1TPnwY1edOTtFPJ"

# 替换为 Qwen3 的 API 端点（假设）
openai.api_base = "https://api.bianxie.ai/v1"

evaluate_form = """
3. 评分标准：
# 评分 20分：已经推导出了24
input:[24],output:20  # 已成功，归为 20 分

# 评分 19~16：输入数字可通过简单加减得到 24,根据复杂度评定16~19
input:[10,14],output:18  # 10+14=24，归为 19~16 分
input:[9,6,9],output:16  # 9+6+9=24，归为 19~16 分
input:[15,9],output:19  # 15+9=24，归为 19~16 分

# 评分 15~12：需要乘除加减才可得到24,根据复杂度评定16~19
input:[6,4], output:15  # 6 × 4 =24，归为 15~12 分
input:[12,2],output:15  # 12 × 2=24，归为 15~12 分

# 评分 11~8：需要复杂操作,即加减乘除和括号都要用上，才能得到24，根据复杂度评定11~8
input:[3，3，7，7],output:8 # 需要复杂操作，归为 11~8 分 (7 × (3 + 3/7))

# 评分 7~2：无法快速推导,根据推理潜力，评定7~2
input:[6, 7, 9],output:5 #  无法快速推导, 归为2分
input:[4, 7, 11],output:2  # 无法快速推导, 归为2分
# 评分 1：无法推导出 24
input:[1,1,1,1],output:1 # 无法推导出24
input:[2,2,2,2])", 1
"""

def generate_dataset(num_samples=10, output_file="24_dataset_patt2_4.txt"):
    dataset = []
    task = get_task("game24")
    print(f"🎯 开始生成 {num_samples} 个样本...")

    for i in range(num_samples):
        # 构造提示词（prompt）引导模型生成符合要求的样本
        prompt = ("""
你是一个24点游戏专家，请生成input:输入和对应的output:评分。
譬如 初始24点输入为四个数：4 5 6 7
中间状态为：
4x5=20,left: [20,5,7]
20-5=15,left: [15,7]
15+7=22,left: [22]
这就是一次推理，现在需要你对可能出现的left，进行评分
!!!!注意 你所有的input 必须能通过原始24点变化得到!!!!
要求：
1. 生成input,input为中间状态集合，output为得分
2. 评分范围为1-20，评分规则如下：
   - 20分：直接得到24
   - 19-16分：剩余数字可通过简单加减得到24
   - 15-12分：需要乘除或括号
   - 11-8分：需要复杂操作（嵌套括号）
   - 7-2分：无法快速推导
   - 1分：无法推导出24

请生成符合要求的样本注意 你的输出为json格式,注意 你的输出一定不能和示例有任何相同,你生成的input应尽量在每个分段涵盖3，2个input输入"""+
"\n本次原始的24点游戏输入是"+task.get_input(i+800)+
""","!!!!注意,你需要先进行24点游戏推理 生成一堆中间input步骤,注意每个数字只能用一次，如4 5 6 7 可生成中间步骤 20（即4×5） 6 7，之后推理24点的数字就变成20 6 7了 你所有的input 必须能通过原始24点加减乘除等变化得到!!!!"\n
然后再对中间步骤进行评分，你的评分不用覆盖全部1-20 只是根据生成的input中间步骤实事求是即可，你的输出格式会被解析为jsonl，不要有任何其他无关输出！！！:
{"output": 15, "input": [12,2],"reason":"需要乘除加减才可得到24,只需一次乘法"},
{},
{}
""")

        try:
            # 调用 OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-all",  # 替换为实际模型名称
                messages=[
                    {"role": "system", "content": "你是一个24点游戏专家，擅长生成中间状态和评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # 控制输出多样性
                max_tokens=3000,
                request_timeout=10000
            )

            # 提取模型输出
            generated_text = response.choices[0].message.content.strip()
            dataset.append(generated_text)

            print(f"[{i + 1}/{num_samples}] 生成成功: {generated_text}")
            time.sleep(0.5)  # 避免频繁请求

        except Exception as e:
            print(f"请求失败: {e}")
            continue
    # 保存到文件
            # 每 5 个样本保存一次
        if (i + 1) % 5 == 0 or (i + 1) == num_samples:
            with open(output_file, "a", encoding="utf-8") as f:
                for line in dataset:
                    f.write(line + "\n")
            print(f"💾 已保存前 {i + 1} 个样本到 {output_file}")
            dataset.clear()  # 清空缓存


    print(f"数据集已保存到 {output_file}")


# 调用函数生成数据集
generate_dataset(num_samples=100)