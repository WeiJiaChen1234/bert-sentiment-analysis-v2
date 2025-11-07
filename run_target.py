import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='BERT 目标域情感分析')
parser.add_argument('--model', type=str, default='bert_target', help='选择模型')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'target_domain'  # 目标域数据集
    model_name = args.model  # bert_target
    x = import_module('models.' + model_name)
    config = x.Config()

    # 设置随机种子，确保可复现
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # 加载数据（自动读取采样后的训练集）
    print("加载目标域情感分析数据...")
    start_time = time.time()
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print(f"数据加载耗时: {time_dif}")
    print(f"训练集样本数: {len(train_data)}, 验证集: {len(dev_data)}, 测试集: {len(test_data)}")

    # 初始化模型
    model = x.Model(config).to(config.device)

    # 打印可训练参数详情（验证解冻成功）
    print("\n可训练参数列表（前10个）:")
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    for name in trainable_params[:10]:
        print(f"  {name}")
    print(f"  ... 共{len(trainable_params)}个可训练参数")

    # 初始化网络权重
    init_network(model)
    print(f"\n{model_name} 模型初始化完成，开始训练...")

    # 启动训练
    train(config, model, train_iter, dev_iter, test_iter, log_step=config.log_step)