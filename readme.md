# 1. CNN의 개념

![CNN 프로세스](./images/cnn_process.jpg)

다음은 input(입력), 기능 학습 및 분류 단계를 보여주는 CNN 알고리즘 프로세스를 설명하는 자세한 다이어그램입니다. 이미지는 컨볼루셔널 신경망(Convolutional Neural Network)이 각 단계를 통해 데이터를 처리하고 분류하는 방법을 시각적으로 나타냅니다.

<br><br><br><br>

# 2. CNN을 활용한 손글씨 판별하기

손글씨로 그리기나 업로드된 이미지의 숫자 판별하기

<br>

## 2-1. 프로젝트 파일 구조

```csharp
handwrite_syatem/
│
├── app.py  # Flask 웹 애플리케이션의 진입점
├── model.py  # CNN 러닝 모델
├── uploads/  # 사용자가 업로드한 파일을 저장하는 디렉터리
├── templates/
│   ├── im.html     # 이미지 업로드 템플릿
│   ├── index.html  # 메인 페이지 템플릿
│   └── index_old.html  # 결과 전 페이지 템플릿
└── mnist_model.h5  # 학습한 내용이 저장되는 파일
```

<br><br><br>

## 2-2. 패키지 설치

```bsah
(.venv) pip install Flask tensorflow numpy pillow
```

<br><br><br>

## 2-3. 소스 코드

### 2-3-1. model.py

```python
#@title MNIST

import tensorflow.keras.utils as utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


X_val = X_train[50000:]
Y_val = Y_train[50000:]
X_train = X_train[:50000]
Y_train = Y_train[:50000]

X_train = X_train.reshape(50000, 784).astype('float32') / 255.0
X_val = X_val.reshape(10000, 784).astype('float32') / 255.0
X_test = X_test.reshape(10000, 784).astype('float32') / 255.0

Y_train = utils.to_categorical(Y_train)
Y_val = utils.to_categorical(Y_val)
Y_test = utils.to_categorical(Y_test)

model = Sequential()
model.add(Dense(units=800, input_dim=28*28, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


hist = model.fit(X_train, Y_train, epochs=5,
                 batch_size=10, validation_data=(X_val, Y_val),verbose=1)

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
```

<br><br>

### 2-3-2. app.py

```python
from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/im')
def im(result=None):
    return render_template('im.html', result=result)

# 설정: 업로드된 파일을 저장할 디렉토리
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# 파일 확장자가 허용된 것인지 확인하는 함수
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def process_file(file_path):
    try:
        with Image.open(file_path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            # img = img.resize((28, 28))
            img_array = np.array(img)
            model = load_model('mnist_model.h5')
            test_data = img_array.reshape(1, 784)
            yhat_test = model.predict(test_data)
            print(np.argmax(yhat_test))
            processed_data = np.argmax(yhat_test)

            return str(processed_data)
    except Exception as e:
        return f"Error processing file: {e}"

@app.route('/upload', methods=['POST'])
def upload():
    data = request.json
    image_data = data.get('image')
    # 이미지 데이터를 처리하는 코드 추가
    # 예: image_data를 리스트로 변환
    image_list = list(map(int, image_data.split(',')))


    model = load_model('mnist_model.h5')
    test_data = np.array(image_list)
    test_data = test_data.reshape(1, 784)
    yhat_test = model.predict(test_data)
    print(np.argmax(yhat_test))

    # 필요한 추가 처리 수행
    processed_data = np.argmax(yhat_test)

    return jsonify({'status': 'success', 'processed_data': int(processed_data)})


@app.route('/upload2', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return im("No file part")
    file = request.files['file']

    # 파일이 없거나 파일명이 비어있는 경우
    if file.filename == '':
        return im("No selected file")

    # 파일이 허용된 형식인지 확인
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 파일을 읽어 처리
        result = process_file(file_path)

        return im(result)

    return im("File type is not allowed")


if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, render_template
#
# app = Flask(__name__)
#
# @app.route('/')
# def home():
#     name = "Alice"
#     value = 42
#     return render_template('index.html', name=name, value=value)
#
# if __name__ == '__main__':
#     app.run(debug=True)
```

<br><br>

### 2-3-3. template/im.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Upload</title>
</head>
<body>
    <h1>Upload a File</h1>
    <form action="/upload2" method="post" enctype="multipart/form-data">
        <input type="file" name="file" />
        <input type="submit" value="Upload" />
    </form>
    {% if result %}
    <h2>Processing Result:</h2>
    <p>{{ result }}</p>
    {% endif %}
</body>
</html>
```

<br><br>

### 2-3-4. tempalte/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Drawing Canvas</title>
</head>
<body>
    <table>
        <td style="border-style: none;">
            <div style="border: solid 2px #666; width: 143px; height: 144px;">
                <canvas width="140" height="140" id="drawingCanvas"></canvas>
            </div>
        </td>
        <td style="border-style: none;">
            <button onclick="clearCanvas()">Clear</button>
            <button onclick="sendData()">Send</button>
        </td>
    </table>
    <div id="result"></div>
    <script type="text/javascript">
        var pixels = [];
        for (var i = 0; i < 28 * 28; i++) pixels[i] = 0;

        var click = 0;

        var canvas = document.getElementById("drawingCanvas");
        canvas.addEventListener("mousemove", function(e) {
            if (e.buttons == 1) {
                click = 1;
                canvas.getContext("2d").fillStyle = "rgb(0,0,0)";
                canvas.getContext("2d").fillRect(e.offsetX, e.offsetY, 8, 8);
                var x = Math.floor(e.offsetY * 0.2);
                var y = Math.floor(e.offsetX * 0.2) + 1;
                for (var dy = 0; dy < 2; dy++) {
                    for (var dx = 0; dx < 2; dx++) {
                        if ((x + dx < 28) && (y + dy < 28)) {
                            pixels[(y + dy) + (x + dx) * 28] = 1;
                        }
                    }
                }
            } else {
                if (click == 1) {
                    click = 0;
                }
            }
        });

        function clearCanvas() {
            canvas.getContext("2d").fillStyle = "rgb(255,255,255)";
            canvas.getContext("2d").fillRect(0, 0, 140, 140);
            for (var i = 0; i < 28 * 28; i++) pixels[i] = 0;
        }

        function sendData() {
            var result = pixels.join(",");
            fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: result })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = 'Processed Data: ' + data.processed_data;
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
```

<br><br>

### 2-3-5. 사용자 디렉토리 만들기

uploads : 판별할 이미지가 업로드되는 디렉토리

<br><br><br>

## 2-4. CNN 모델 훈련

<br><br><br>

## 2-5. 애플리케이션 실행

```bash
(.venv) python app.py runserver
```

<br><br><br><br>

# 3. CNN을 활용한 이미지 분류 및 정보 유추하기



<br><br><br><br>

# 4. CNN을 활용한 이미지 채점하기



<br><br><br><br>

# 5. CNN을 활용한 이미지 생성기 제작하기

## 5-1. 프로젝트 파일 구조

먼저, 프로젝트의 파일 구조를 정의합니다.

```csharp
face_generation_app/
│
├── app.py  # Flask 웹 애플리케이션의 진입점
├── static/
│   └── uploads/  # 사용자가 업로드한 파일을 저장하는 디렉터리
├── templates/
│   ├── index.html  # 메인 페이지 템플릿
│   └── result.html  # 결과 페이지 템플릿
├── models/
│   ├── cnn_model.py  # CNN 모델 정의
│   ├── gan_model.py  # GAN, DCGAN, StyleGAN 모델 정의
│   └── model_weights/  # 사전 학습된 모델 가중치 저장
└── requirements.txt  # 필요한 Python 패키지 목록
```

<br><br><br>

## 5-2. 프로젝트 프로세스

### 5-2-1. 기본적인 웹 애플리케이션 구성

Flask 설정: 웹 애플리케이션의 기본적인 라우팅과 템플릿 렌더링 기능을 설정합니다.
HTML 템플릿: index.html과 result.html을 사용하여 사용자 인터페이스를 구성합니다.
파일 업로드: 사용자가 업로드한 이미지를 static/uploads/ 디렉터리에 저장합니다.

<br><br>

### 5-2-2. 모델 구축 및 로드
CNN 모델: 입력된 이미지나 데이터를 처리할 CNN 모델을 정의하고, 이를 사용하여 전처리된 데이터를 생성합니다.
GAN 모델: GAN, DCGAN, StyleGAN과 같은 다양한 GAN 모델을 정의하고, 사전 학습된 가중치를 로드하여 입력된 데이터로부터 얼굴 이미지를 생성합니다.

<br><br>

### 5-2-3. 이미지 생성 및 결과 반환
이미지 생성: 업로드된 이미지나 프롬프트를 기반으로 GAN 모델을 사용해 새로운 얼굴 이미지를 생성합니다.
결과 페이지: 생성된 이미지를 결과 페이지에 표시하고, 사용자가 다운로드할 수 있도록 제공합니다.

<br><br><br>

## 5-3. 소스 코드

### 5-3-1. app.py

```python
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from models.cnn_model import preprocess_input
from models.gan_model import generate_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 이미지 생성
        generated_image_path = generate_image(file_path)
        
        return render_template('result.html', 
                               original_image=file_path, 
                               generated_image=generated_image_path)

if __name__ == "__main__":
    app.run(debug=True)
```

<br><br>

### 5-3-2. models/cnn_model.py

```python
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def preprocess_input(image_path):
    model = VGG16(weights='imagenet', include_top=False)
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
```

<br><br>

### 5-3-3. models/gan_model.py

```python
import os
import numpy as np
from tensorflow.keras.models import load_model

# GAN, DCGAN, StyleGAN 모델 불러오기
gan_model = load_model('models/model_weights/gan_model.h5')
dcgan_model = load_model('models/model_weights/dcgan_model.h5')
stylegan_model = load_model('models/model_weights/stylegan_model.h5')

def generate_image(input_image_path):
    # 예시: 입력 이미지를 GAN으로 변환하여 새로운 이미지를 생성
    latent_vector = np.random.normal(size=(1, 100))
    generated_image = gan_model.predict(latent_vector)
    
    generated_image_path = os.path.join('static/uploads/', 'generated_image.png')
    generated_image.save(generated_image_path)
    
    return generated_image_path
```

<br><br>

### 5-3-4. templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Generation</title>
</head>
<body>
    <h1>Generate a Face Image</h1>
    <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
        <label for="file">Upload an image:</label>
        <input type="file" name="file">
        <input type="submit" value="Generate">
    </form>
</body>
</html>
```

<br><br>

### 5-3-5. templates/result.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Generation Result</title>
</head>
<body>
    <h1>Generated Face Image</h1>
    <p>Original Image:</p>
    <img src="{{ url_for('static', filename=original_image.split('/')[-1]) }}" alt="Original Image">
    <p>Generated Image:</p>
    <img src="{{ url_for('static', filename=generated_image.split('/')[-1]) }}" alt="Generated Image">
</body>
</html>
```

<br><br><br>

## 5-4. requirements.txt

```bsah
pip install Flask tensorflow numpy pillow
```

<br><br><br>

## 5-5. 프로젝트 설정 및 실행

프로젝트 디렉터리 설정: 위에 명시된 파일 구조로 디렉터리를 설정합니다.
필요한 패키지 설치: requirements.txt 파일을 통해 필요한 Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

모델 파일 준비: models/model_weights/ 폴더에 사전 학습된 GAN 모델 가중치를 다운로드하여 배치합니다.
서버 실행: Flask 애플리케이션을 실행합니다.

```bash
python app.py
```

웹 브라우저에서 접근: 로컬 호스트(http://127.0.0.1:5000/)에서 애플리케이션에 접근하여 이미지를 업로드하고 생성된 얼굴 이미지를 확인합니다.

