import torch
from models.bert_source import Config  # 导入你的配置类

# 初始化配置，查看设备是否为GPU
config = Config()
# 验证CUDA是否可用
print("CUDA可用状态:", torch.cuda.is_available())
print("当前设备:", config.device)
print("GPU型号（若有）:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无")