import os
import torch
from transformers import BertModel, BertTokenizerFast


class Config(object):
    """目标域（情感分析）模型配置类（适配2万条训练集）"""

    def __init__(self):
        # 1. 基础配置
        self.model_name = "bert_target"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"目标域模型使用设备: {self.device}")

        # 2. 数据路径（使用采样后的训练集）
        self.project_root = os.path.abspath(".")
        self.data_dir = os.path.join(self.project_root, "target_domain")
        self.train_path = os.path.join(self.data_dir, "train_sampled.txt")  # 采样后的训练集
        self.dev_path = os.path.join(self.data_dir, "dev.txt")
        self.test_path = os.path.join(self.data_dir, "test.txt")
        self.class_path = os.path.join(self.data_dir, "class.txt")

        # 3. 迁移学习核心配置
        self.source_model_path = os.path.join(self.project_root, "saved_dict", "bert_source.ckpt")
        self.bert_path = os.path.join(self.project_root, "bert_pretrain")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.bert_path)
        self.hidden_size = 768

        # 4. 训练参数（适配2万条数据，快速收敛）
        self.class_list = [x.strip() for x in open(self.class_path, 'r', encoding='utf-8').readlines()]
        self.num_classes = len(self.class_list)
        self.num_epochs = 5
        self.batch_size = 4  # 适配MX350显存
        self.pad_size = 15
        self.learning_rate = 3e-6  # 稳定微调，避免过拟合
        self.require_improvement = 1000  # 早停阈值，适配小数据量
        self.log_step = 50  # 保持日志频率
        self.early_stop = True

        # 模型保存路径
        self.save_path = os.path.join(self.project_root, "saved_dict", f"{self.model_name}.ckpt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


class Model(torch.nn.Module):
    """目标域BERT模型（解冻3层，适配小数据量）"""

    def __init__(self, config):
        super(Model, self).__init__()
        # 1. 加载BERT基础模型
        self.bert = BertModel.from_pretrained(config.bert_path, output_hidden_states=False)

        # 2. 加载源域BERT权重（验证加载）
        try:
            source_model_state = torch.load(config.source_model_path, map_location=config.device)
            bert_state_dict = {k.replace("bert.", ""): v for k, v in source_model_state.items() if
                               k.startswith("bert.")}
            missing_keys, unexpected_keys = self.bert.load_state_dict(bert_state_dict, strict=False)
            print(f"\n源域权重加载状态：")
            print(f"  缺失参数（前5个）: {missing_keys[:5]}")
            print(f"  意外参数（前5个）: {unexpected_keys[:5]}")
            if len(missing_keys) > 500:
                print("⚠️  警告：源域权重加载可能失败，请检查source_model_path是否正确！")
        except Exception as e:
            print(f"\n❌ 源域权重加载失败：{str(e)}")
            print("⚠️  将使用BERT基础预训练权重继续训练")

        # 3. 解冻最后3层BERT（9-11层）+ 分类头
        # 冻结所有BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False

        # 解冻最后3层
        frozen_layers = [9, 10, 11]
        for layer_num in frozen_layers:
            layer = self.bert.encoder.layer[layer_num]
            for param in layer.parameters():
                param.requires_grad = True
            print(f"✅ 已解冻BERT层: encoder.layer.{layer_num}")

        # 4. 分类头（强制可训练）
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)
        torch.nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.0)
        self.fc.weight.requires_grad = True
        self.fc.bias.requires_grad = True
        print("✅ 已解冻分类头参数: fc.weight, fc.bias")

        # 迁移到设备
        self.to(config.device)
        self.device = config.device

    def forward(self, x):
        token_ids, _, mask = x
        token_ids = token_ids.to(self.device)
        mask = mask.to(self.device)
        _, pooled = self.bert(input_ids=token_ids, attention_mask=mask, return_dict=False)
        out = self.fc(pooled)
        return out