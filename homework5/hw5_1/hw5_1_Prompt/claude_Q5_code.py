import matplotlib.pyplot as plt
import numpy as np

# 準備數據
epochs = range(20, 101, 20)  # [20, 40, 60, 80, 100]
folds_data = {
    'Fold 1': {
        'train_loss': [0.8513, 0.6563, 0.5641, 0.5081, 0.4210],
        'val_loss': [0.7916, 0.5603, 0.4587, 0.3833, 0.3108],
        'val_acc': [0.7333, 0.8333, 0.9000, 0.9333, 0.9333]
    },
    'Fold 2': {
        'train_loss': [0.8727, 0.6820, 0.5635, 0.5202, 0.4240],
        'val_loss': [0.8468, 0.6292, 0.5182, 0.4410, 0.3692],
        'val_acc': [0.8333, 0.9667, 0.9667, 0.9667, 0.9667]
    },
    'Fold 3': {
        'train_loss': [0.8593, 0.7048, 0.6049, 0.5375, 0.4943],
        'val_loss': [0.8891, 0.6461, 0.5422, 0.4829, 0.4263],
        'val_acc': [0.7000, 0.8000, 0.8333, 0.8333, 0.9333]
    },
    'Fold 4': {
        'train_loss': [0.8733, 0.7067, 0.5777, 0.5449, 0.4422],
        'val_loss': [0.9282, 0.7061, 0.5791, 0.4509, 0.4213],
        'val_acc': [0.6667, 0.7667, 0.8667, 0.8667, 0.9000]
    },
    'Fold 5': {
        'train_loss': [0.8416, 0.6695, 0.5750, 0.4666, 0.3797],
        'val_loss': [0.9086, 0.6973, 0.5849, 0.4870, 0.3980],
        'val_acc': [0.7000, 0.8333, 0.8000, 0.9000, 0.9667]
    }
}

# 創建圖表
plt.style.use('seaborn')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# 繪製訓練損失
for i, (fold, data) in enumerate(folds_data.items()):
    ax1.plot(epochs, data['train_loss'], 'o-', label=fold, color=colors[i])
ax1.set_title('Training Loss per Fold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 繪製驗證損失
for i, (fold, data) in enumerate(folds_data.items()):
    ax2.plot(epochs, data['val_loss'], 'o-', label=fold, color=colors[i])
ax2.set_title('Validation Loss per Fold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# 繪製驗證準確率
for i, (fold, data) in enumerate(folds_data.items()):
    ax3.plot(epochs, data['val_acc'], 'o-', label=fold, color=colors[i])
ax3.set_title('Validation Accuracy per Fold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.legend()
ax3.grid(True)

# 調整布局
plt.tight_layout()

# 保存圖表
plt.savefig('iris_training_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# 計算並繪製最終結果的箱形圖
final_accuracies = [data['val_acc'][-1] for data in folds_data.values()]

plt.figure(figsize=(8, 6))
plt.boxplot(final_accuracies)
plt.title('Distribution of Final Validation Accuracies Across Folds')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('iris_final_accuracies_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
