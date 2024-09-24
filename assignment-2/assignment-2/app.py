from flask import Flask, render_template, request, jsonify
import base64
import io
from matplotlib import pyplot as plt
import numpy as np
import random

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def kmeans():
    if request.method == 'POST':
       plot_url=generate_new()
       return plot_url
    else:
        plot_url = generate_new()
        return render_template('index.html',plot_url=plot_url)

def generate_new():
    x = np.random.uniform(-10, 10, 300)
    y = np.random.uniform(-10, 10, 300)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1)  # horizontal line (x-axis) at y=0
    plt.axvline(0, color='black', linewidth=1)  # vertical line (y-axis) at x=0
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Convert plot to PNG image and encode in base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Return base64 image data as a plain string response
    return plot_url

if __name__ == '__main__':
    app.run(debug=True,port=3000)