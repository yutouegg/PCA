from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gevent import pywsgi

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

file_path = None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global file_path
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        return jsonify({'columns': columns})


@app.route('/columns', methods=['GET'])
def get_columns():
    global file_path
    if file_path:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        return jsonify({'columns': columns})
    else:
        return jsonify({'error': 'No file uploaded yet'}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    global file_path
    if not file_path:
        return jsonify({'error': 'No file uploaded yet'}), 400

    data = request.json
    target = data.get('target')

    df = pd.read_csv(file_path)
    X, y = data_change(df, target)
    explained_variance_df = pca_app(X, y)
    image_path = save_plot(explained_variance_df)

    table_data = explained_variance_df.to_dict(orient='records')
    static_image_path = os.path.join('static', 'pca_plot.png')

    return jsonify({"table": table_data, "image_path": static_image_path}), 200


def data_change(df, target):
    cols = df.columns
    columns = [col for col in cols if col != target] + [target]
    df = df[columns]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_standardized = (X - X.mean()) / X.std()

    return X_standardized, y


def pca_app(X, y):
    pca = PCA()
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_

    explained_variance_df = pd.DataFrame({
        '特征': [f'{col}' for col in X.columns[:]],
        '占比': explained_variance_ratio
    })

    return explained_variance_df


def save_plot(df):
    plt.rcParams['font.family'] = 'SimHei'
    labels = df['特征']
    plt.pie(df['占比'], labels=labels, autopct='%.2f%%')
    plt.title('各特征占比')
    image_path = os.path.join(app.config['STATIC_FOLDER'], 'pca_plot.png')
    plt.savefig(image_path)
    plt.close()
    return image_path

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 7000), app)
    server.serve_forever()

