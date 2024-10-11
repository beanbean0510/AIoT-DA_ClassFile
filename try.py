from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Generate synthetic data and plot it
def generate_data(a, b, noise, num_points):
    np.random.seed(42)
    x = np.random.rand(num_points) * 10
    y = a * x + b + np.random.randn(num_points) * noise
    return x, y

# Plotting function to show the regression line and points
def plot_data(x, y, a, b):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, a * x + b, color='red', label=f'Regression Line: y={a}x+{b}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get user input from form
    a = float(request.form['a'])
    b = float(request.form['b'])
    noise = float(request.form['noise'])
    num_points = int(request.form['num_points'])

    # Generate data and plot
    x, y = generate_data(a, b, noise, num_points)
    plot_url = plot_data(x, y, a, b)

    # Return the plot and equation
    return jsonify({'plot_url': plot_url, 'equation': f'y = {a}x + {b}'})

if __name__ == '__main__':
    app.run(debug=True)