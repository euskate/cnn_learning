import base64
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model import load_model, predict_marks

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
GRADED_FOLDER = 'static/graded/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADED_FOLDER'] = GRADED_FOLDER

# 폴더 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADED_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 모델 로드
model = load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send_image', methods=['GET', 'POST'])
def send_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('pointing_page', filename=filename))
    return render_template('send_image.html')


@app.route('/pointing_page/<filename>', methods=['GET', 'POST'])
def pointing_page(filename):
    if request.method == 'POST':
        student_name = request.form.get('student_name')
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 모델을 사용하여 이미지 채점 및 결과 저장
        score, grading_info, graded_filename = predict_marks(model, img_path)

        # 채점 결과 저장
        result = {
            "student_name": student_name,
            "uploaded_image": filename,
            "graded_image": graded_filename,
            "score": score,
            "grading_info": grading_info
        }
        result_path = os.path.join(app.config['GRADED_FOLDER'], f'{filename.split(".")[0]}.json')
        with open(result_path, 'w') as json_file:
            json.dump(result, json_file)

        return redirect(url_for('result_page', result_json=f'{filename.split(".")[0]}.json'))

    return render_template('pointing_page.html', filename=filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/graded/<filename>')
def graded_file(filename):
    return send_from_directory(app.config['GRADED_FOLDER'], filename)


@app.route('/result/<result_json>')
def result_page(result_json):
    result_path = os.path.join(app.config['GRADED_FOLDER'], result_json)
    with open(result_path, 'r') as json_file:
        result = json.load(json_file)
    return render_template('result_page.html', result=result)


def predict_grades(model, image_path):
    # 이미지 로드 및 전처리
    img = Image.open(image_path)
    img = img.resize((64, 64))  # 모델 입력 크기에 맞게 조정
    img_array = np.array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    # 모델 예측
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # 예시 점수 및 등급 할당
    score = 100  # 기본 점수
    grading_info = {
        "color": ("X", "red"),
        "icon": ("V", "blue"),
        "typography": ("/", "yellow"),
        "layout": ("X", "green")
    }

    # 모델 예측에 따라 점수 조정
    if predicted_class == 0:  # 예: 'X'일 경우
        score -= 10
    elif predicted_class == 1:  # 예: 'V'일 경우
        score -= 5
    elif predicted_class == 2:  # 예: '/'일 경우
        score -= 1

    # 이미지에 주석 추가
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 예시로 X, V, /를 랜덤 위치에 추가 (실제 구현 시 적절한 위치 및 주석 추가)
    draw.text((10, 10), "X", fill="red")
    draw.text((50, 50), "V", fill="blue")
    draw.text((90, 90), "/", fill="yellow")

    # 채점된 이미지 저장
    graded_filename = f'{os.path.splitext(os.path.basename(image_path))[0]}_graded.png'
    graded_path = os.path.join(app.config['GRADED_FOLDER'], graded_filename)
    img.save(graded_path)

    return score, grading_info, graded_filename


if __name__ == '__main__':
    app.run(debug=True)
