import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 設定隨機種子
torch.manual_seed(42)

# 載入資料
iris = load_iris()
X = iris.data
y = iris.target

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 資料標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 轉換為 PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 定義模型
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 初始化模型、損失函數和優化器
model = IrisNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 設定 TensorBoard
writer = SummaryWriter('runs/iris_experiment')

# 訓練模型
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向傳播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向傳播
    loss.backward()
    optimizer.step()
    
    # 計算準確率
    _, predicted = torch.max(outputs.data, 1)
    train_accuracy = (predicted == y_train).sum().item() / y_train.size(0)
    
    # 記錄到 TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}')

# 評估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    test_accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'\nTest Accuracy: {test_accuracy:.4f}')

writer.close()
