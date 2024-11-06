### 載入套件
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

### 從 Kaggle 載入資料集
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
# gender_submission.csv 是繳交範例資料，不會用到因此不載入

### 查看資料集狀況
print(test_data.info(), "\n")
print(train_data.info(), "\n")

### 資料探勘與前處理
# 將 test data 和 train data 合併
total_data = pd.concat([test_data, train_data], ignore_index = True)
print(total_data.info())

# PassengerId
train_data['PassengerId'].head(15)

# Survived
total_data['Survived'].unique()
sns.countplot(train_data['Survived'])

# Pclass
total_data['Pclass'].unique()
sns.countplot(train_data['Pclass'])
sns.barplot(x = 'Pclass', y = "Survived", data = train_data)

# Sex
total_data['Sex'].unique()
sns.countplot(train_data['Sex'])
labelencoder = LabelEncoder()
total_data['Sex'] = labelencoder.fit_transform(total_data['Sex'])
total_data['Sex'].unique()

# Age
train_data['Age_qcut'] = pd.qcut(train_data['Age'], 8)
train_data['Age_qcut'].unique()
sns.barplot(x = train_data['Age_qcut'], y = train_data['Survived'])
sns.histplot(train_data['Age'])
total_data.loc[total_data['Age'] <= 16,'Age'] = 1
total_data.loc[total_data['Age'] != 1,'Age'] = 2
total_data['Age'].unique()

#SibSp
sns.countplot(train_data['SibSp'])
sns.barplot(x = 'SibSp', y = 'Survived', data=train_data)
total_data.loc[ (total_data['SibSp']==1) | (total_data['SibSp']==2), 'SibSp'] = 1
total_data.loc[ total_data['SibSp'] > 2,'SibSp'] = 2
total_data.loc[ total_data['SibSp'] < 1 ,'SibSp'] = 0
total_data['SibSp'].value_counts()

# Parch
sns.countplot(train_data['Parch'])
total_data['Parch'].value_counts()
sns.barplot(x = 'Parch', y = 'Survived', data = train_data)
total_data['Parch_cut'] = pd.cut(total_data['Parch'], [-1, 0, 3, 9])
sns.countplot(total_data['Parch_cut'])
sns.barplot(x = total_data['Parch_cut'], y = total_data['Survived'])
total_data.loc[(total_data['Parch'] > 0) & (total_data['Parch'] <= 3), 'Parch'] = 2
total_data.loc[total_data['Parch'] > 4 ,'Parch'] = 4
total_data.loc[total_data['Parch'] < 1, 'Parch' ] = 1
total_data['Parch'].value_counts()

# Fare
sns.histplot(train_data[train_data['Survived'] == 1]['Fare'])
sns.histplot(train_data[train_data['Survived'] == 0]['Fare'],color = 'red')
train_data['Fare_cut'] = pd.cut(train_data['Fare'], [-1, 15, 50, 1000])
sns.countplot(train_data['Fare_cut'])
sns.barplot(x = train_data['Fare_cut'], y = train_data['Survived'])
total_data.loc[total_data['Fare'] <= 15,'Fare'] = 1
total_data.loc[(total_data['Fare'] > 15) & (total_data['Fare'] <= 50), 'Fare'] = 2
total_data.loc[total_data['Fare'] > 2, 'Fare'] = 3
total_data['Fare'].value_counts()
total_data['Fare'] = total_data['Fare'].fillna(1)

# Embarked
total_data['Embarked'].unique()
sns.countplot(total_data['Embarked'])
sns.barplot(x = train_data['Embarked'], y = train_data['Survived'])
total_data['Embarked'] = total_data['Embarked'].fillna('S')
total_data['Embarked'].unique()
total_data['Embarked'] = labelencoder.fit_transform(total_data['Embarked'])
total_data['Embarked'].value_counts()

### 模型訓練
# 將DataFram轉成ndarray
train_x = total_data[total_data['Survived'].notnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
train_y = total_data[total_data['Survived'].notnull()][['Survived']].values
test_x = total_data[total_data['Survived'].isnull()][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values

# 做正規化
minmax = MinMaxScaler()
train_x = minmax.fit_transform(train_x)
test_x = minmax.transform(test_x)

train_x = torch.tensor(train_x, dtype = torch.float32)
train_y = torch.tensor(train_y, dtype = torch.float32)
test_x = torch.tensor(test_x, dtype = torch.float32)

# 分成train_set、validate_set
train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.2)

class dataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.n_sample = len(x)
  def __getitem__(self,index):
    return self.x[index], self.y[index]
  def __len__(self):
    return self.n_sample
  
train_set = dataset(train_x, train_y)

# DataLoader
train_loader = DataLoader(dataset = train_set, batch_size = 100, shuffle = True)

# Model
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(train_x.shape[1], 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
        
    )
  def forward(self, x):
    return self.net(x)
model = Model()

critirion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
epoch = 2000
n_batch = len(train_loader)
best_acc = 0

for i in range(epoch):
  for j, (samples, labels) in enumerate(train_loader):
    pre = model(samples)
    loss = critirion(pre, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'epoch = {i+1}/{epoch}, batch = {j+1}/{n_batch}, loss = {loss:.4f}  ' ,end='')
    with torch.no_grad():
      n_sample = train_x.shape[0]
      pre = model(train_x)
      pre = pre.round()
      n_correct = (pre == train_y).sum()
      train_acc = n_correct / n_sample
      print(f'train_acc = {train_acc:.4f}  ',end='')
    with torch.no_grad():
      n_sample = valid_x.shape[0]
      pre = model(valid_x)
      pre = pre.round()
      n_correct = (pre == valid_y).sum()
      valid_acc = n_correct / n_sample
      print(f'valid_acc = {valid_acc:.4f}'  )
      if(best_acc < valid_acc and (abs(valid_acc - train_acc)) < 0.01 and (valid_acc < train_acc) and (train_acc < 0.825)):
        best_acc = valid_acc #更新最好正確率
        torch.save(model, "model.pth") #儲存model
        print("update") #輸出update字樣

best_model_pre_cabin = torch.load("model.pth")

with torch.no_grad():
      n_sample = valid_x.shape[0]
      pre = best_model_pre_cabin(valid_x)
      pre = pre.round()
      n_correct = (pre == valid_y).sum()
      acc = n_correct / n_sample
      print(f'valid_acc = {acc:.4f}'  )

with torch.no_grad():
      n_sample = train_x.shape[0]
      pre = best_model_pre_cabin(train_x)
      pre = pre.round()
      n_correct = (pre == train_y).sum()
      acc = n_correct / n_sample
      print(f'train_acc = {acc:.4f}'  )

with torch.no_grad():
      n_sample = valid_x.shape[0]
      pre = best_model_pre_cabin(test_x)
      pre = pre.round()
      pre = pre.view(-1).numpy().astype(np.int32)
      answer = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : pre})

answer.to_csv('IT_submission.csv', index = False)

### 顯示混淆矩陣
with torch.no_grad():
    # 只在驗證集上進行預測
    pre = model(valid_x)
    pre = pre.round()  # 將預測結果四捨五入為 0 或 1
    pre = pre.view(-1).numpy().astype(int)  # 轉換為 1D numpy array 並轉換為整數
    valid_y = valid_y.view(-1).numpy().astype(int)  # 轉換驗證集標籤為 1D numpy array

    # 確保預測結果與驗證標籤大小一致
    print(f"預測結果 shape: {pre.shape}")
    print(f"驗證標籤 shape: {valid_y.shape}")

    # 計算混淆矩陣
    cm = confusion_matrix(valid_y, pre)

    # 顯示混淆矩陣
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.ylabel('實際值')
    plt.xlabel('預測值')
    plt.title('混淆矩陣')
    plt.show()
