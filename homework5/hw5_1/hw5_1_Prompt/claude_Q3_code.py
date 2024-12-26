import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

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
        self.layer1 = nn.Linear(4, 32)  # 減少神經元數量
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 增加 dropout 比例
        self.bn1 = nn.BatchNorm1d(32)   # 添加 Batch Normalization
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
fold_results = []

# 準備 TensorBoard
writer = SummaryWriter('runs/iris_experiment_with_kfold')

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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 添加 L2 正則化
    
    # 訓練模型
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # 訓練模式
        model.train()
        optimizer.zero_grad()
        
        # 前向傳播
        train_outputs = model(X_train)
        loss = criterion(train_outputs, y_train)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        # 評估模式
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            
            # 計算準確率
            _, predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (predicted == y_val).sum().item() / y_val.size(0)
            
            # 記錄到 TensorBoard
            writer.add_scalar(f'Loss/train_fold_{fold}', loss.item(), epoch)
            writer.add_scalar(f'Loss/val_fold_{fold}', val_loss.item(), epoch)
            writer.add_scalar(f'Accuracy/val_fold_{fold}', val_accuracy, epoch)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, '
                  f'Val Accuracy: {val_accuracy:.4f}')
    
    # 儲存當前 fold 的結果
    fold_results.append(val_accuracy)

# 輸出整體結果
mean_accuracy = np.mean(fold_results)
std_accuracy = np.std(fold_results)
print(f'\nK-fold 交叉驗證結果:')
print(f'平均準確率: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
print(f'各 fold 準確率: {[f"{acc:.4f}" for acc in fold_results]}')

writer.close()
