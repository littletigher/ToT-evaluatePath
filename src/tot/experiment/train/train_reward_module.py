#!/usr/bin/env python3
"""
奖励模型训练脚本 - PEFT优化版本
使用方法:
python reward_model_train_no_wandb.py \
    --base_model Qwen/Qwen2-0.5B-Instruct \
    --dataset_path D:/Github/trl/datasets/HPD_EDP \
    --output_dir ./Qwen2-0.5B-2-Reward-LoRA \
    --max_length 2048 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3
"""

import os
import math
import torch
import argparse
import numpy as np
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    PeftConfig
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练奖励模型")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                        help="基础模型名称")
    parser.add_argument("--dataset_path", type=str, default="D:\\Github\\trl\\datasets\\HPD_EDP",
                        help="数据集路径")
    parser.add_argument("--output_dir", type=str, default="./Qwen2-0.5B-3-Reward-LoRA",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--use_8bit", action="store_true",
                        help="是否使用8位量化")
    parser.add_argument("--eval_steps", type=int, default=200,
                        help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存间隔步数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录间隔步数")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj",
                        help="LoRA目标模块，用逗号分隔")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="从checkpoint恢复训练的路径")
    return parser.parse_args()


def format_conversation(messages):
    """将对话列表格式化为单个字符串"""
    if isinstance(messages, list):
        conversation = ""
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    conversation += f"Human: {content}\n\n"
                elif role == "assistant":
                    conversation += f"Assistant: {content}\n\n"
                else:
                    conversation += f"{role}: {content}\n\n"
        return conversation.strip()
    elif isinstance(messages, str):
        return messages
    else:
        return str(messages)


def prepare_dataset(dataset, tokenizer, max_length):
    """准备数据集，将chosen和rejected文本转换为模型输入"""
    print(dataset['chosen'][0])

    def tokenize_function(examples):
        if isinstance(examples["chosen"], list) and len(examples["chosen"]) > 0:
            chosen_texts = [format_conversation(item) for item in examples["chosen"]]
        else:
            chosen_texts = [format_conversation(examples["chosen"])]

        if isinstance(examples["rejected"], list) and len(examples["rejected"]) > 0:
            rejected_texts = [format_conversation(item) for item in examples["rejected"]]
        else:
            rejected_texts = [format_conversation(examples["rejected"])]

        # 标记化chosen文本
        chosen_tokens = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 标记化rejected文本
        rejected_tokens = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # 组合成模型所需格式
        model_inputs = {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
            "labels": torch.ones(len(chosen_texts), dtype=torch.float)
        }

        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


class RewardDataCollator:
    """自定义数据整理器，处理成对的数据"""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        input_ids_chosen = torch.stack(
            [example["input_ids_chosen"] if isinstance(example["input_ids_chosen"], torch.Tensor)
             else torch.tensor(example["input_ids_chosen"]) for example in examples])
        attention_mask_chosen = torch.stack(
            [example["attention_mask_chosen"] if isinstance(example["attention_mask_chosen"], torch.Tensor)
             else torch.tensor(example["attention_mask_chosen"]) for example in examples])
        input_ids_rejected = torch.stack(
            [example["input_ids_rejected"] if isinstance(example["input_ids_rejected"], torch.Tensor)
             else torch.tensor(example["input_ids_rejected"]) for example in examples])
        attention_mask_rejected = torch.stack(
            [example["attention_mask_rejected"] if isinstance(example["attention_mask_rejected"], torch.Tensor)
             else torch.tensor(example["attention_mask_rejected"]) for example in examples])

        batch = {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
            "labels": torch.ones(len(examples), dtype=torch.float)
        }

        return batch


def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            gpu_metrics.update({
                f'gpu_{i}_memory_used': torch.cuda.memory_allocated(i) / (1024 ** 3),  # 转换为GB
                f'gpu_{i}_memory_cached': torch.cuda.memory_reserved(i) / (1024 ** 3),  # 转换为GB
                f'gpu_{i}_memory_free': (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(
                    i)) / (1024 ** 3)  # 转换为GB
            })
        return gpu_metrics
    return {'gpu_memory': 'N/A'}


def get_cpu_memory_usage():
    """获取CPU内存使用情况"""
    return {
        'cpu_memory_percent': psutil.virtual_memory().percent,
        'cpu_memory_used_gb': psutil.virtual_memory().used / (1024 ** 3),  # 转换为GB
        'cpu_memory_free_gb': psutil.virtual_memory().free / (1024 ** 3)  # 转换为GB
    }


def log_system_metrics():
    """记录系统指标"""
    metrics = {
        **get_gpu_memory_usage(),
        **get_cpu_memory_usage(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    logging.info(f"系统指标: {metrics}")
    return metrics


class RewardTrainer(Trainer):
    """自定义训练器，用于处理奖励模型的成对比较训练"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.last_metrics_log = time.time()
        self.metrics_interval = 60  # 每60秒记录一次系统指标

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids_chosen = inputs["input_ids_chosen"]
        attention_mask_chosen = inputs["attention_mask_chosen"]
        input_ids_rejected = inputs["input_ids_rejected"]
        attention_mask_rejected = inputs["attention_mask_rejected"]

        # 获取chosen文本的分数
        chosen_outputs = model(
            input_ids=input_ids_chosen,
            attention_mask=attention_mask_chosen
        )
        chosen_rewards = chosen_outputs.logits.squeeze(-1)

        # 获取rejected文本的分数
        rejected_outputs = model(
            input_ids=input_ids_rejected,
            attention_mask=attention_mask_rejected
        )
        rejected_rewards = rejected_outputs.logits.squeeze(-1)

        # 计算损失 - 我们希望chosen的分数高于rejected的分数
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        if return_outputs:
            return loss, (chosen_outputs, rejected_outputs)
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """自定义评估方法，计算偏好准确率"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        metrics = {}
        self.model.eval()

        all_chosen_rewards = []
        all_rejected_rewards = []

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with torch.no_grad():
                chosen_outputs = self.model(
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"]
                )
                chosen_rewards = chosen_outputs.logits.squeeze(-1).cpu().numpy()

                rejected_outputs = self.model(
                    input_ids=batch["input_ids_rejected"],
                    attention_mask=batch["attention_mask_rejected"]
                )
                rejected_rewards = rejected_outputs.logits.squeeze(-1).cpu().numpy()

                all_chosen_rewards.extend(chosen_rewards)
                all_rejected_rewards.extend(rejected_rewards)

        accuracy = (np.array(all_chosen_rewards) > np.array(all_rejected_rewards)).mean()
        metrics[f"{metric_key_prefix}_accuracy"] = float(accuracy)

        mean_diff = np.mean(np.array(all_chosen_rewards) - np.array(all_rejected_rewards))
        metrics[f"{metric_key_prefix}_mean_diff"] = float(mean_diff)

        metrics[f"{metric_key_prefix}_loss"] = float(self.compute_loss(
            self.model,
            {
                "input_ids_chosen": batch["input_ids_chosen"],
                "attention_mask_chosen": batch["attention_mask_chosen"],
                "input_ids_rejected": batch["input_ids_rejected"],
                "attention_mask_rejected": batch["attention_mask_rejected"]
            }
        ).item())

        return metrics


def load_datasets(dataset_path):
    """加载数据集"""
    try:
        if os.path.isdir(dataset_path):
            dataset = load_dataset(dataset_path)
        else:
            dataset = load_dataset(dataset_path)

        splits = list(dataset.keys())

        if "train" not in splits:
            raise ValueError("数据集必须包含train分割")

        if "validation" not in splits and "test" not in splits:
            dataset = dataset["train"].train_test_split(test_size=0.1)

        return dataset
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        return None


def main():
    args = parse_args()
    start_time = time.time()

    # 记录初始系统状态
    initial_metrics = log_system_metrics()
    logging.info("🚀 开始训练奖励模型...")
    logging.info(f"📁 基础模型: {args.base_model}")
    logging.info(f"📁 数据集路径: {args.dataset_path}")
    logging.info(f"📁 输出目录: {args.output_dir}")
    logging.info(f"🔢 最大长度: {args.max_length}")
    logging.info(f"💾 批处理大小: {args.batch_size}")
    logging.info(f"📈 学习率: {args.learning_rate}")
    logging.info(f"🔄 训练轮数: {args.num_train_epochs}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置量化参数
    if args.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 配置LoRA
    target_modules = args.target_modules.split(",")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None
    )

    # 准备模型进行训练
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model)

    # 应用LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 确保PAD token id设置正确
    model.config.pad_token_id = tokenizer.pad_token_id

    # 加载数据集
    logging.info("\n📊 正在加载数据集...")
    dataset = load_datasets(args.dataset_path)

    if dataset is None:
        print("❌ 数据集加载失败，退出训练")
        return

    # 准备数据集
    logging.info("🔄 正在准备数据集...")
    train_dataset = prepare_dataset(dataset["train"], tokenizer, args.max_length)

    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = prepare_dataset(dataset["validation"], tokenizer, args.max_length)
    elif "test" in dataset:
        eval_dataset = prepare_dataset(dataset["test"], tokenizer, args.max_length)

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True if eval_dataset else False,
        greater_is_better=True,
        fp16=True,
        logging_dir=f"{args.output_dir}/logs",
        report_to=[],  # 不使用任何报告工具
        remove_unused_columns=False
    )

    # 创建自定义数据整理器
    data_collator = RewardDataCollator(tokenizer, args.max_length)

    # 初始化训练器
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    logging.info("🚂 开始训练...")
    if args.resume_from_checkpoint:
        logging.info(f"🔄 从checkpoint恢复训练: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # 保存最终模型
    logging.info("💾 保存模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 计算总用时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 记录最终统计信息
    final_metrics = log_system_metrics()
    final_stats = {
        'total_training_time_hours': hours,
        'total_training_time_minutes': minutes,
        'total_training_time_seconds': seconds,
        'initial_metrics': initial_metrics,
        'final_metrics': final_metrics
    }

    logging.info(f"训练完成，统计信息: {final_stats}")
    logging.info("🎉 训练完成!")


if __name__ == "__main__":
    main()