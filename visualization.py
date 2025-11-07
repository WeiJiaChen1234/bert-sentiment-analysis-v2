import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows用黑体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac用这个
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建保存文件夹
os.makedirs('figures', exist_ok=True)

# ---------------------- 1. 训练收敛曲线（核心！证明模型收敛）----------------------
# 从你的日志中提取的关键数据（Iter、Train Loss、Val Loss、Val Acc、Val F1）
iterations = [0, 50, 100, 200, 300, 400, 550, 700, 850, 1400, 1600]
train_loss = [0.8861, 0.8528, 1.2711, 0.8514, 0.6984, 0.7083, 0.5597, 0.3531, 0.7113, 0.5268, 0.6426]
val_loss = [0.9829, 0.9106, 0.7730, 0.7249, 0.6878, 0.6565, 0.6226, 0.6150, 0.6620, 0.5685, 0.5692]
val_acc = [49.99, 49.26, 47.24, 48.69, 54.93, 61.39, 66.22, 64.41, 60.81, 69.02, 70.26]
val_f1 = [0.0027, 0.0095, 0.1410, 0.5435, 0.5810, 0.6322, 0.6557, 0.6912, 0.6995, 0.7002, 0.7300]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 左图：Loss曲线
ax1.plot(iterations, train_loss, label='训练损失', color='#FF6B6B', linewidth=2, marker='o', markersize=4)
ax1.plot(iterations, val_loss, label='验证损失', color='#4ECDC4', linewidth=2, marker='s', markersize=4)
ax1.set_xlabel('迭代次数（Iter）', fontsize=11)
ax1.set_ylabel('损失值（Loss）', fontsize=11)
ax1.set_title('模型训练收敛曲线（Loss）', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 右图：Acc/F1曲线
ax2.plot(iterations, val_acc, label='验证准确率（%）', color='#45B7D1', linewidth=2, marker='o', markersize=4)
ax2.plot(iterations, [f1*100 for f1 in val_f1], label='验证F1分数（×100）', color='#96CEB4', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('迭代次数（Iter）', fontsize=11)
ax2.set_ylabel('百分比（%）', fontsize=11)
ax2.set_title('模型性能提升曲线（Acc/F1）', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/训练收敛曲线.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 2. 类别性能对比图（展示两类识别效果）----------------------
categories = ['负面情感（negative）', '正面情感（positive）']
precision = [0.7414, 0.6604]  # 精确率
recall = [0.5952, 0.7913]     # 召回率
f1_score = [0.6603, 0.7200]   # F1分数

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision, width, label='精确率', color='#FF9F43', alpha=0.8)
rects2 = ax.bar(x, recall, width, label='召回率', color='#10AC84', alpha=0.8)
rects3 = ax.bar(x + width, f1_score, width, label='F1分数', color='#5F27CD', alpha=0.8)

# 添加数值标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

ax.set_xlabel('情感类别', fontsize=11)
ax.set_ylabel('性能指标值', fontsize=11)
ax.set_title('两类情感识别性能对比', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/类别性能对比图.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 3. 混淆矩阵热力图（专业替代文字矩阵）----------------------
cm = np.array([[3561, 2422], [1242, 4710]])  # 你的混淆矩阵数据
class_names = ['负面情感（negative）', '正面情感（positive）']

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={'fontsize': 12, 'fontweight': 'bold'})

ax.set_xlabel('预测类别', fontsize=11)
ax.set_ylabel('真实类别', fontsize=11)
ax.set_title('情感分类混淆矩阵', fontsize=12, fontweight='bold')

# 添加总样本数和准确率标注
total_samples = cm.sum()
accuracy = cm.trace() / total_samples
ax.text(0.5, -0.15, f'总样本数：{total_samples} | 准确率：{accuracy:.3f}',
        transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/混淆矩阵热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 4. 参数配置可视化（体现工程优化）----------------------
params = {
    '可训练参数数': '302万',
    '解冻BERT层数': '3层（9-11层）',
    '训练数据量': '8000条（平衡）',
    'Batch Size': '4',
    '学习率': '3e-6',
    '训练总耗时': '2.5小时',
    '早停阈值': '1000步'
}

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # 隐藏坐标轴

# 创建表格
table_data = [[k, v] for k, v in params.items()]
table = ax.table(cellText=table_data,
                 colLabels=['优化参数', '配置值'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.4, 0.6])

# 美化表格
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置表头样式
for i in range(2):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置内容样式
for i in range(1, len(table_data)+1):
    for j in range(2):
        table[(i, j)].set_facecolor('#F8F9FA')
        if j == 0:
            table[(i, j)].set_text_props(weight='bold')

ax.set_title('模型工程优化参数配置', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/参数配置可视化.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 所有可视化图表已生成完成！保存路径：figures/")
print("包含：")
print("1. 训练收敛曲线.png（核心证明收敛）")
print("2. 类别性能对比图.png（两类效果对比）")
print("3. 混淆矩阵热力图.png（专业混淆矩阵）")
print("4. 参数配置可视化.png（工程优化说明）")