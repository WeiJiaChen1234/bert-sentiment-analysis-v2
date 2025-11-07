# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from transformers import AdamW, get_linear_schedule_with_warmup


def init_network(model, method='xavier', exclude='embedding', seed=123):
    torch.manual_seed(seed)
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, log_step=100, early_stop=True):
    start_time = time.time()
    model.train()

    # 1. 检查可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\n训练参数检查:")
    print(f"  可训练参数总数: {trainable_params} (正常应≈300万)")
    print(f"  总参数总数: {all_params}")
    if trainable_params < 100000:
        print("⚠️  可训练参数过少！模型可能无法收敛，请检查解冻逻辑")

    # 2. 优化器配置（适配小数据量）
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.001  # 轻度权重衰减，防止过拟合
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    # 3. 优化器+调度器（快速收敛）
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=1e-8)
    total_steps = len(train_iter) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 4. 训练状态变量
    total_batch = 0
    dev_best_loss = float('inf')
    dev_best_f1 = -1
    last_improve = 0
    flag = False
    gradient_accumulation_steps = 1  # 关闭累积，快速迭代

    for epoch in range(config.num_epochs):
        print(f'\nEpoch [{epoch + 1}/{config.num_epochs}]')
        epoch_loss = 0.0

        for i, (trains, labels) in enumerate(train_iter):
            trains = tuple(t.to(model.device) for t in trains)
            labels = labels.to(model.device)

            # 前向传播
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # 日志打印
            if total_batch % log_step == 0:
                true = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                train_acc = metrics.accuracy_score(true, predic)
                train_f1 = metrics.f1_score(
                    true, predic,
                    average='binary' if config.num_classes == 2 else 'macro',
                    zero_division=0
                )

                # 验证集评估
                dev_acc, dev_loss, dev_f1 = evaluate(config, model, dev_iter)

                # 保存最优模型
                if (dev_f1 > dev_best_f1) or (dev_loss < dev_best_loss and dev_f1 >= dev_best_f1):
                    dev_best_loss = dev_loss
                    dev_best_f1 = dev_f1
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                # 打印日志
                time_dif = get_time_dif(start_time)
                msg = (f'Iter: {total_batch:>6},  Train Loss: {loss.item():>5.4f},  '
                       f'Train Acc: {train_acc:>6.2%},  Train F1: {train_f1:>6.4f},  Val Loss: {dev_loss:>5.4f},  '
                       f'Val Acc: {dev_acc:>6.2%},  Val F1: {dev_f1:>6.4f},  Time: {time_dif} {improve}')
                print(msg)
                model.train()

            total_batch += 1

            # 早停判断
            if early_stop and (total_batch - last_improve > config.require_improvement):
                print(f"超过{config.require_improvement}步无优化，自动停止训练...")
                flag = True
                break

        if flag:
            break
        print(f'Epoch {epoch + 1} 平均训练损失: {epoch_loss / len(train_iter):.4f}')

    # 测试集评估
    test(config, model, test_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path, map_location=model.device))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, test_f1 = evaluate(config, model, test_iter, test=True)
    msg = f'Test Loss: {test_loss:>5.4f},  Test Acc: {test_acc:>6.2%},  Test F1: {test_f1:>6.4f}'
    print(msg)
    print("\nPrecision, Recall and F1-Score (按类别):")
    print(test_report)
    print("Confusion Matrix:")
    print(test_confusion)
    print(f"测试耗时: {get_time_dif(start_time)}")


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    with torch.no_grad():
        for texts, labels in data_iter:
            texts = tuple(t.to(model.device) for t in texts)
            labels = labels.to(model.device)
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.item()

            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    f1 = metrics.f1_score(
        labels_all, predict_all,
        average='binary' if config.num_classes == 2 else 'macro',
        zero_division=0
    )

    if test:
        report = metrics.classification_report(
            labels_all, predict_all,
            target_names=config.class_list,
            digits=4,
            zero_division=0
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion, f1
    return acc, loss_total / len(data_iter), f1