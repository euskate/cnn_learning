from flask import Flask, request, render_template
import boto3
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

rekognition = boto3.client('rekognition')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text')
def index2():
    return render_template('index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        with open(file_path, 'rb') as image:
            response = rekognition.detect_labels(Image={'Bytes': image.read()})

        labels = response['Labels']
        return render_template('index.html', labels=labels)

@app.route('/upload2', methods=['POST'])
def upload2():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        with open(file_path, 'rb') as image:
            response = rekognition.detect_text(Image={'Bytes': image.read()})

        labels = response['TextDetections']
        return render_template('index2.html', labels=labels)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)


