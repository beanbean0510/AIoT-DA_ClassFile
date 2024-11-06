from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# 初始化 Flask app
app = Flask(__name__)

# 根據使用者輸入產生合成資料的函數
def generate_data(a, b, noise, num_points):
    # 隨機產生 seed 
    np.random.seed(42)
    # 生成隨機的 x 值 (0 - 10)
    x = np.random.rand(num_points) * 10
    # 生成 y 值，利用 y = ax + b + noise
    y = a * x + b + np.random.randn(num_points) * noise
    return x, y

# 繪製所產生的資料和迴歸線的函數
def plot_data(x, y, a, b):
    # 設定圖片大小
    plt.figure(figsize=(8, 6))
    # 產生數據點的散點圖
    plt.scatter(x, y, color='blue', label='Data Points')
    # 畫出迴歸線 (y = ax + b)
    plt.plot(x, a * x + b, color='red', label=f'Regression Line: y={a}x+{b}')
    # 在兩軸加上標籤
    plt.xlabel('X')
    plt.ylabel('Y')
    # 顯示圖例（用於識別資料點與迴歸線）
    plt.legend()
    plt.tight_layout()

    # 將繪圖儲存到記憶體中
    img = io.BytesIO()
    plt.savefig(img, format='png')
    # 重設
    img.seek(0)
    # 將圖片轉換為 base64，以便可以在 HTML 中顯示
    plot_url = base64.b64encode(img.getvalue()).decode()
    # 關閉圖片以釋出記憶體
    plt.close()
    return plot_url

# 設定前往網站的路線
@app.route('/')
def index():
    # 渲染 HTML 範本 (index.html)
    return render_template('index.html')

# 處理表單提交並產生回歸資料
@app.route('/generate', methods=['POST'])
def generate():
    # 從表單取得使用者輸入（POST 請求）
    a = float(request.form['a'])                   # Slope (a)
    b = float(request.form['b'])                   # Intercept (b)
    noise = float(request.form['noise'])           # Noise level
    num_points = int(request.form['num_points'])   # Number of data points

    # 根據使用者輸入產生合成數據
    x, y = generate_data(a, b, noise, num_points)
    # 繪製資料並取得圖像 URL（base64 編碼）
    plot_url = plot_data(x, y, a, b)

    # Return a JSON response containing the plot URL and the equation
    return jsonify({'plot_url': plot_url, 'equation': f'y = {a}x + {b}'})

# Run the app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)     # Enable debug mode for easier development