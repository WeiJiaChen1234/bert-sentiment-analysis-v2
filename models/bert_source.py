import os
import torch
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """源域（新闻分类）模型配置类，处理路径兼容问题"""

    def __init__(self):
        # 1. 基础配置
        self.model_name = "bert_source"  # 模型名称，用于保存权重
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备（GPU/CPU）
        print(f"当前使用设备: {self.device}")  # 新增：打印设备信息，方便确认是否使用GPU

        # 2. 数据路径配置（关键：用os.path处理路径，适配Windows）
        self.project_root = os.path.abspath(".")  # 项目根目录
        self.data_dir = os.path.join(self.project_root, "source_domain")
        self.train_path = os.path.join(self.data_dir, "train.txt")
        self.dev_path = os.path.join(self.data_dir, "dev.txt")
        self.test_path = os.path.join(self.data_dir, "test.txt")
        self.class_path = os.path.join(self.data_dir, "class.txt")

        # 3. 模型参数
        self.bert_path = os.path.join(self.project_root, "bert_pretrain")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768  # BERT隐藏层维度

        # 4. 类别与训练参数（适配MX350 2GB显存，调整batch_size避免OOM）
        self.class_list = [x.strip() for x in open(self.class_path, 'r', encoding='utf-8').readlines()]
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 8  # 关键修改：从16降至8，适配2GB显存（若仍报错可改为4）
        self.pad_size = 20  # 关键修改：从32缩短至20，减少单条数据显存占用
        self.learning_rate = 5e-5
        self.require_improvement = 1000
        self.save_path = os.path.join(self.project_root, "saved_dict", f"{self.model_name}.ckpt")
        # 新增：确保保存目录存在，避免保存模型时报错
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)


class Model(torch.nn.Module):
    """源域BERT模型（用于新闻分类，适配GPU训练）"""

    def __init__(self, config):
        super(Model, self).__init__()
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 解冻BERT所有参数（源域数据充足时微调）
        for param in self.bert.parameters():
            param.requires_grad = True
        # 分类头
        self.fc = torch.nn.Linear(config.hidden_size, config.num_classes)

        # 关键修改：将模型迁移到指定设备（GPU/CPU）
        self.to(config.device)
        # 记录设备，方便forward中使用
        self.device = config.device

    def forward(self, x):
        # x的结构：(token_ids, seq_len, mask)
        # 关键修改：确保输入数据在模型同一设备（防止设备不匹配报错）
        context = x[0].to(self.device)  # token_ids迁移到模型设备
        mask = x[2].to(self.device)  # 注意力掩码迁移到模型设备

        # BERT输出：(last_hidden_state, pooler_output)
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # 分类输出
        out = self.fc(pooled)
        return out