########## Homework 3-1 1D compare to  logistic regression with SVM on simple case ##########
# CRISP-DM Steps Implementation for Logistic Regression and SVM (RBF Kernel)

### Step 1: Business Understanding
# Goal: Classify binary outcomes using logistic regression and SVM on 300 random variables to analyze classification performance and decision boundaries.

### Step 2: Data Understanding
# 導入必要的套件
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### Step 3: Data Preparation
# 設置隨機種子，確保結果可重現
np.random.seed(42)

# 生成300個0到1000之間的隨機數，並重塑為二維數組（-1表示自動計算行數，1表示1列）
X = np.random.uniform(0, 1000, 300).reshape(-1, 1)
# 定義二元結果：當 X 值在 500 到 800 之間時，Y = 1；否則 Y = 0
# ravel():用於將數組展平為一維
Y = np.where((X > 500) & (X < 800), 1, 0).ravel()

### Step 4: Modeling

# 訓練邏輯回歸模型
logistic_model = LogisticRegression()  # 創建邏輯回歸模型實例
logistic_model.fit(X, Y)  # 訓練模型
Y1 = logistic_model.predict(X)  # 使用邏輯回歸模型進行預測


# 使用RBF核的SVM模型（用於非線性分類）
svm_rbf_model = SVC(kernel='rbf')  # 創建SVM模型實例，使用RBF核
svm_rbf_model.fit(X, Y)  # 訓練模型
Y2_rbf = svm_rbf_model.predict(X)  # 使用SVM模型進行預測

# 計算決策邊界用於可視化
x_range = np.linspace(0, 1000, 1000).reshape(-1, 1)  # 創建均勻分佈的點用於繪製決策邊界
# 計算邏輯回歸的決策邊界（預測概率>=0.5的點）
logistic_boundary = logistic_model.predict_proba(x_range)[:, 1] >= 0.5
# 計算SVM的決策邊界
svm_rbf_boundary = svm_rbf_model.decision_function(x_range)

### Step 5: Evaluation
# 評估模型在訓練數據上的準確率
logistic_accuracy = accuracy_score(Y, Y1)  # 計算邏輯回歸的準確率
svm_rbf_accuracy = accuracy_score(Y, Y2_rbf)  # 計算SVM的準確率

# 輸出準確率結果
print(f"邏輯回歸準確率: {logistic_accuracy:.2f}")
print(f"SVM (RBF核) 準確率: {svm_rbf_accuracy:.2f}")

### Step 6: Deployment (Visualization of Results)
plt.figure(figsize=(14, 6))

# 繪製邏輯回歸的結果
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y1, 'ro', label='Logistic Predictions', markersize=2)
plt.plot(x_range, logistic_boundary, 'k--', label='Logistic Decision Boundary')
plt.title("Logistic Regression Classification")
plt.xlabel("X")
plt.ylabel("Y / Y1")
plt.legend()

# 繪製SVM的結果
plt.subplot(1, 2, 2)
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y2_rbf, 'go', label='SVM (RBF) Predictions', markersize=2)
plt.plot(x_range, (svm_rbf_boundary >= 0).astype(int), 'k--', label='SVM (RBF) Decision Boundary')
plt.title("SVM Classification with RBF Kernel")
plt.xlabel("X")
plt.ylabel("Y / Y2")
plt.legend()

plt.tight_layout()
plt.show()

########## Homework 3-2 2D SVM with streamlit deployment (3D plot) -dataset 分布在feature plane上圓形 ##########
########## Homework 3-3 2D dataset 分布在feature plane上非圓形 ##########

from flask import Flask, render_template, request
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def generate_circular_data(n_samples=300, noise=0.1):
    """生成圓形分布的數據"""
    np.random.seed(42)
    radius = np.random.uniform(0, 1, n_samples)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    
    X1 = radius * np.cos(theta) + np.random.normal(0, noise, n_samples)
    X2 = radius * np.sin(theta) + np.random.normal(0, noise, n_samples)
    
    Y = (radius < 0.5).astype(int)
    
    return np.column_stack((X1, X2)), Y

def generate_moon_data(n_samples=300, noise=0.1):
    """生成月牙形數據"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def create_svm_plots(n_samples=300, noise=0.1, kernel='rbf', C=1.0):
    """創建兩種數據分布的SVM模型和圖形"""
    # 創建子圖
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('圓形分布', '月牙形分布')
    )
    
    # 處理圓形數據
    X_circle, y_circle = generate_circular_data(n_samples, noise)
    scaler_circle = StandardScaler()
    X_circle_scaled = scaler_circle.fit_transform(X_circle)
    
    # 處理月牙形數據
    X_moon, y_moon = generate_moon_data(n_samples, noise)
    scaler_moon = StandardScaler()
    X_moon_scaled = scaler_moon.fit_transform(X_moon)
    
    # 訓練模型並創建可視化
    models = []
    accuracies = []
    
    for i, (X, y) in enumerate([(X_circle_scaled, y_circle), (X_moon_scaled, y_moon)]):
        # 訓練SVM模型
        model = SVC(kernel=kernel, C=C)
        model.fit(X, y)
        models.append(model)
        accuracies.append(model.score(X, y))
        
        # 創建網格
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # 計算決策函數
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 添加決策邊界平面
        fig.add_trace(
            go.Surface(x=xx, y=yy, z=Z, opacity=0.5,
                      colorscale='RdBu', name=f'Decision Boundary {i+1}'),
            row=1, col=i+1
        )
        
        # 添加數據點
        fig.add_trace(
            go.Scatter3d(x=X[y==0, 0], y=X[y==0, 1], z=np.zeros(np.sum(y==0)),
                        mode='markers', marker=dict(size=5, color='blue'),
                        name=f'Class 0 - {i+1}'),
            row=1, col=i+1
        )
        
        fig.add_trace(
            go.Scatter3d(x=X[y==1, 0], y=X[y==1, 1], z=np.zeros(np.sum(y==1)),
                        mode='markers', marker=dict(size=5, color='red'),
                        name=f'Class 1 - {i+1}'),
            row=1, col=i+1
        )
    
    # 更新布局
    fig.update_layout(
        title=f'SVM分類可視化 ({kernel} kernel)',
        height=600,
        width=1200,
        showlegend=True,
    )
    
    for i in range(1, 3):
        fig.update_scenes(
            dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Decision Function'
            ),
            row=1, col=i
        )
    
    return fig, accuracies

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 從表單獲取參數
        kernel = request.form.get('kernel', 'rbf')
        C = float(request.form.get('C', 1.0))
        n_samples = int(request.form.get('n_samples', 300))
        noise = float(request.form.get('noise', 0.1))
        
        # 創建圖形
        fig, accuracies = create_svm_plots(n_samples, noise, kernel, C)
        plot_html = fig.to_html(full_html=False)
        
        return render_template('index.html', 
                             plot=plot_html,
                             accuracy_circle=f'{accuracies[0]*100:.2f}%',
                             accuracy_moon=f'{accuracies[1]*100:.2f}%',
                             kernel=kernel,
                             C=C,
                             n_samples=n_samples,
                             noise=noise)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)