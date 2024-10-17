from flask import Flask, render_template
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    # 讀取資料
    data = pd.read_csv('D:\Class\GPT class\homework1\hw1_2\data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # 轉換 y 列為數值型，並處理缺失值
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(inplace=True)  # 刪除缺失值

    # Feature Selection: Adding lagged features
    for lag in range(1, 6):  # Create lagged features for the last 5 days
        data[f'y_lag_{lag}'] = data['y'].shift(lag)
    data.dropna(inplace=True)  # Drop rows with NaN values from lagging

    # Auto Regression 模型
    y = data['y']
    X = data[[f'y_lag_{lag}' for lag in range(1, 6)]]  # Select lagged features
    model = sm.OLS(y, sm.add_constant(X)).fit()  # Use OLS regression for evaluation

    # 預測未來值
    predictions = []
    last_lags = data.iloc[-1][[f'y_lag_{lag}' for lag in range(1, 6)]].values  # Get last lags as an array
    last_lags = np.concatenate(([1], last_lags))  # Add a constant term (1) at the beginning

    for i in range(5):  # Predict the next 5 values
        pred = model.predict(last_lags)  # Predict using the model
        predictions.append(pred[0])  # Append the prediction
        # Update last_lags for the next prediction
        last_lags = np.roll(last_lags[1:], 1)  # Shift left (excluding the constant)
        last_lags[0] = pred[0]  # Update the first lag with the current prediction
        last_lags = np.concatenate(([1], last_lags))  # Add constant again

    # Model Evaluation
    actual = y[-5:].values
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mse = mean_squared_error(actual, predictions)  # Calculate MSE
    r_squared = model.rsquared  # Get the R² value from the fitted model

    # 繪製圖表
    fig, ax = plt.subplots()
    ax.plot(y.index, y, label='Actual')
    ax.plot(pd.date_range(y.index[-1], periods=6, freq='D')[1:], predictions, label='Predicted', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('y values')
    ax.legend()
    ax.set_title(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}, MSE: {mse:.2f}, R²: {r_squared:.2f}')  # Display all metrics

    # 保存圖表到內存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    plt.close(fig)  # 關閉圖表以釋放內存

    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
