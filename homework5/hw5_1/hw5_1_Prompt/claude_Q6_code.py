import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 設定隨機種子
torch.manual_seed(42)
np.random.seed(42)

# 載入資料
iris = load_iris()
X = iris.data
y = iris.target

# 定義模型
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(16)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        return x

# 設定 K-fold 交叉驗證
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# 創建一個更好組織的 TensorBoard 記錄目錄
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'runs/iris_experiment_{current_time}'
writer = SummaryWriter(log_dir)

# 添加超參數記錄
hp_config = {
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'dropout_rate': 0.3,
    'weight_decay': 0.01,
    'k_folds': k_folds,
    'model_architecture': 'IrisNet (32-16-3)',
}
# 將配置寫入 TensorBoard
writer.add_text('Hyperparameters', str(hp_config))

# 為模型結構創建示例輸入
example_input = torch.randn(1, 4)
model = IrisNet()
writer.add_graph(model, example_input)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'\nFold {fold + 1}/{k_folds}')
    
    # 準備資料
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 資料標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # 轉換為 PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    
    # 初始化模型、損失函數和優化器
    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 訓練模型
    num_epochs = 100
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        # 訓練模式
        model.train()
        optimizer.zero_grad()
        
        # 前向傳播
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        
        # 計算訓練準確率
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_accuracy = (train_predicted == y_train).sum().item() / y_train.size(0)
        
        # 反向傳播
        train_loss.backward()
        optimizer.step()
        
        # 評估模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            # 計算驗證準確率
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
            
            # 記錄到 TensorBoard
            writer.add_scalars(f'Loss/fold_{fold+1}', {
                'train': train_loss.item(),
                'validation': val_loss.item()
            }, epoch)
            
            writer.add_scalars(f'Accuracy/fold_{fold+1}', {
                'train': train_accuracy,
                'validation': val_accuracy
            }, epoch)
            
            # 記錄權重和梯度的分布
            for name, param in model.named_parameters():
                writer.add_histogram(f'fold_{fold+1}/{name}', param.data, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'fold_{fold+1}/{name}.grad', param.grad, epoch)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')
    
    # 儲存當前 fold 的結果
    fold_results.append(best_val_accuracy)
    
    # 添加每個 fold 的最終性能指標
    writer.add_hparams(
        {'fold': fold},
        {
            'best_accuracy': best_val_accuracy,
            'final_train_loss': train_loss.item(),
            'final_val_loss': val_loss.item()
        }
    )

# 輸出整體結果
mean_accuracy = np.mean(fold_results)
std_accuracy = np.std(fold_results)
print(f'\nK-fold 交叉驗證結果:')
print(f'平均準確率: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
print(f'各 fold 準確率: {[f"{acc:.4f}" for acc in fold_results]}')

# 關閉 TensorBoard writer
writer.close()

print(f'\n訓練完成！可以使用以下指令查看視覺化結果：')
print(f'tensorboard --logdir={log_dir}')
