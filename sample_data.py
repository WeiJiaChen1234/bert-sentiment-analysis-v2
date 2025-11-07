import random
import os
import numpy as np

# 设置随机种子，确保采样可复现
random.seed(123)
np.random.seed(123)


def sample_data(input_path, output_path, sample_size=20000, balance=True):
    """
    随机采样训练集，保持类别平衡
    :param input_path: 原始训练集路径
    :param output_path: 采样后训练集路径
    :param sample_size: 总采样数量（默认2万条）
    :param balance: 是否保持类别平衡（默认是）
    """
    # 读取原始数据，按类别分组
    class_0 = []  # negative（标签0）
    class_1 = []  # positive（标签1）

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按\t分割文本和标签
            try:
                content, label = line.split('\t', 1)
                label = int(label)
                if label == 0:
                    class_0.append(line)
                elif label == 1:
                    class_1.append(line)
            except Exception as e:
                print(f"跳过异常行：{line}，错误：{e}")

    # 计算每类采样数量（保持平衡）
    if balance:
        sample_per_class = min(sample_size // 2, len(class_0), len(class_1))
        sampled_0 = random.sample(class_0, sample_per_class)
        sampled_1 = random.sample(class_1, sample_per_class)
        # 合并并打乱
        sampled_lines = sampled_0 + sampled_1
        random.shuffle(sampled_lines)
    else:
        # 不保持平衡，直接随机采样
        all_lines = class_0 + class_1
        sampled_lines = random.sample(all_lines, min(sample_size, len(all_lines)))

    # 保存采样后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines([line + '\n' for line in sampled_lines])

    # 打印采样信息
    print(f"采样完成！")
    print(f"原始数据：negative {len(class_0)} 条，positive {len(class_1)} 条")
    print(f"采样后数据：共 {len(sampled_lines)} 条，negative {len(sampled_0)} 条，positive {len(sampled_1)} 条")
    print(f"保存路径：{output_path}")


if __name__ == '__main__':
    # 采样目标域训练集
    train_input = os.path.join("target_domain", "train.txt")
    train_output = os.path.join("target_domain", "train_sampled.txt")

    # 确保目标域目录存在
    os.makedirs("target_domain", exist_ok=True)

    sample_data(train_input, train_output, sample_size=8000)