import os
from pathlib import Path
from glob import glob
from swanlab.integration.transformers import SwanLabCallback
import swanlab
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from utils import geo

# 设置环境变量以优化CUDA内存分配
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
base_path = Path('/nfs/dataset-ofs-494-1/project/user/donghaoniu/') # 整个实验的base路径

# 加载分词器与模型
checkpoint_path = base_path.joinpath("checkpoints", "pt2", "checkpoint-4000","sft","checkpoint-134000") # 此处修改实验载入的模型，不区分pt和sft
output_path = Path('/nfs/dataset-ofs-494-1/project/user/donghaoniu/checkpoints/cxp_test') # 实验模型的保存路径，会递归保存在base/checkpoint_path/output_path目录下
model = AutoModelForCausalLM.from_pretrained( # 加载模型
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" # 该参数可选，隔壁组验证说短序列时flash_attention效果不佳，待验证
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path) # 加载tokenizer
# tokenizer.padding_side = 'left'
tokenizer.padding_side = 'right'  # right padding更适合sft训练

# 加载JSONL轨迹数据集
jsonl_dir = base_path.joinpath("data", "geohash_jsonl") # 路径数据base/jsonl_dir

jsonl_files = []
# 可以取消注释需要的日期范围
# jsonl_files.extend(jsonl_dir.glob("202405[2-3][0-9]_geohash.jsonl"))
# jsonl_files.extend(jsonl_dir.glob("202406[0-3][0-9]_geohash.jsonl"))
# jsonl_files.extend(jsonl_dir.glob("202407[0-3][0-9]_geohash.jsonl"))
# jsonl_files.extend(jsonl_dir.glob("202408[0-3][0-9]_geohash.jsonl"))
jsonl_files.extend(jsonl_dir.glob("2024090[1-9]_geohash.jsonl"))
jsonl_files.extend(jsonl_dir.glob("2024091[0-9]_geohash.jsonl"))
jsonl_files.extend(jsonl_dir.glob("2024102[1-9]_geohash.jsonl"))
jsonl_files.extend(jsonl_dir.glob("2024112[0-9]_geohash.jsonl"))
jsonl_files.extend(jsonl_dir.glob("20240920_geohash.jsonl"))

# test日期 # 测试时可注释上面使用下面
# jsonl_files.extend(jsonl_dir.glob("20242121_geohash.jsonl"))

# 加载数据集并进行预处理
dataset = load_dataset("json", data_files=[str(f) for f in jsonl_files], split="train")
dataset = dataset.shuffle(seed=42)

    
def formatting_prompts_func(example):
    trajectory = example["traj"]
    if isinstance(trajectory, str):
        geohash_list = trajectory.split()
    elif isinstance(trajectory, list):
        geohash_list = trajectory
    
    # geohash_list = trajectory.split()  # 将轨迹分割为geohash列表
    # end_start = geohash_list[-2:] + geohash_list[:2] # end_start 作为prompt 尽量与pt阶段保持一致，否则会加大sft时间，后续如果从头开始，建议end+start
    end_start = geohash_list[:2] + geohash_list[-2:] # start+end 作为prompt
    # geohash_list = geo.remove_every_three(geohash_list) # 每三个geo取首个
   # geohash_list = geo.random_remove_one_third(geohash_list) # 每三个geo随机取一个
    human_text = " ".join(end_start)
    human_text = geo.split_geo3(human_text)
    # gpt_text = trajectory  # 完整轨迹作为gpt输出
    gpt_text = " ".join(geohash_list)  # 完整轨迹作为gpt输出
    gpt_text = geo.split_geo3(gpt_text)
    text = f"<|im_start|>{human_text}<|im_end|><|im_start|><|im_start|>{gpt_text}<|im_end|>"  # 按照指定格式组合文本
    return text


# 数据整理器
response_template = "<|im_start|><|im_start|>"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, mlm=False)

swanlab.login(api_key="FMBI9lCfkRQd17jQqD7rv") # 注意修改一下，此处为niudonghao的api，可以将swanlab部署在本地上方便观察实验结果，尽量采用swanlab方式
swanlab_callback = SwanLabCallback(project="Qwen2SFT") # swanlab的项目名称修改，同名称的会归结到同一项目中，直观对比实验结果
# 训练参数配置
training_args = SFTConfig(
    output_dir=str(output_path),
    overwrite_output_dir=True,
    learning_rate=1e-5,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    per_device_train_batch_size=32, # batch_size
    gradient_accumulation_steps=16, # 梯度累积
    save_steps=1_0, # 每10步保存一个模型，此处保存频率过低，建议200-500
    save_total_limit=3, # 最多保存3个模型权重，多余的会先进先出删除掉
    bf16=True,
    logging_steps=20, # 日志记录频率
    max_seq_length=300, # 最大token长度，不压缩设置1000，压缩需要设置300
    packing=False,
    dataset_num_proc=16,
)

# 初始化Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    callbacks=[swanlab_callback],
)

print(dataset[0]["text"])

# 开始训练
trainer.train()
trainer.save_model()  # 保存模型
tokenizer.save_pretrained(output_path)  # 保存分词器
