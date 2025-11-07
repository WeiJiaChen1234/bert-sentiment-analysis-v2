# coding: UTF-8
import torch
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    # 打印数据类别分布（辅助排查不平衡问题）
    def print_class_distribution(data, data_name):
        labels = [item[1] for item in data]
        class_counts = np.bincount(labels)
        print(f"{data_name}集类别分布：{dict(zip(range(len(class_counts)), class_counts))}")

    print_class_distribution(train, "训练")
    print_class_distribution(dev, "验证")
    print_class_distribution(test, "测试")

    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, is_train=False):
        self.batch_size = batch_size
        self.batches = batches
        self.device = device
        self.is_train = is_train  # 标记是否为训练集（仅训练集用加权采样）

        # 训练集：使用加权随机采样，解决批次类别不平衡
        if self.is_train and len(batches) > 0:
            labels = [item[1] for item in batches]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts  # 少数类权重更高
            sample_weights = [class_weights[label] for label in labels]

            # 构建加权采样器
            self.sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(batches),
                replacement=True  # 允许重复采样，确保批次类别均衡
            )
            self.data_loader = torch.utils.data.DataLoader(
                batches,
                batch_size=batch_size,
                sampler=self.sampler,
                collate_fn=self._collate_fn
            )
            self.iter_loader = iter(self.data_loader)
        else:
            # 验证/测试集：保持原有顺序，不采样
            self.n_batches = len(batches) // batch_size
            self.residue = len(batches) % self.n_batches != 0
            self.index = 0

    def _collate_fn(self, datas):
        """自定义拼接函数，与原有逻辑一致"""
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.is_train:
            try:
                return next(self.iter_loader)
            except StopIteration:
                # 一轮结束后重新初始化迭代器
                self.iter_loader = iter(self.data_loader)
                return next(self.iter_loader)
        else:
            # 验证/测试集原有逻辑
            if self.residue and self.index == self.n_batches:
                batches = self.batches[self.index * self.batch_size:]
                self.index += 1
                return self._collate_fn(batches)
            elif self.index >= self.n_batches:
                self.index = 0
                raise StopIteration
            else:
                batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
                self.index += 1
                return self._collate_fn(batches)

    def __iter__(self):
        return self

    def __len__(self):
        if self.is_train:
            return len(self.data_loader)
        else:
            return self.n_batches + 1 if self.residue else self.n_batches


def build_iterator(dataset, config):
    """构建迭代器，训练集启用加权采样，验证/测试集禁用"""
    is_train = 'train' in config.train_path and dataset is build_dataset(config)[0]
    return DatasetIterater(dataset, config.batch_size, config.device, is_train=is_train)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))