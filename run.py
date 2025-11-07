import warnings
# 屏蔽PyTorch的过时API警告
warnings.filterwarnings("ignore", category=UserWarning, message="This overload of add_ is deprecated")
import torch
from models.bert_source import Config, Model
import os
import time  # 新增：导入时间模块
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

# 选择源域模型（对应models/bert_source.py）
model_name = "bert_source"
x = import_module(f"models.{model_name}")

# 初始化源域配置
config = x.Config()

# 设置随机种子，保证结果可复现
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

# 加载源域数据（修复时间计算逻辑）
print("加载源域数据...")
start_time = time.time()  # 记录开始时间
train_data, dev_data, test_data = build_dataset(config)
train_iter = build_iterator(train_data, config)
dev_iter = build_iterator(dev_data, config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)  # 传入开始时间计算耗时
print(f"数据加载完成，耗时：{time_dif}")

# 初始化模型并移动到设备（GPU/CPU）
model = x.Model(config).to(config.device)

# 训练源域模型（复用项目的train_eval.py中的train函数）
from train_eval import train
train(config, model, train_iter, dev_iter, test_iter)