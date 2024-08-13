from model import build_model, train_model

# 학습할 데이터 디렉토리 경로 설정
train_data_dir = 'data/train'
validation_data_dir = 'data/valdi'

# 모델 구축
model = build_model()

# 모델 학습
train_model(model, train_data_dir, validation_data_dir)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 하이퍼파라미터 설정
input_shape = (64, 64, 3)
num_classes = 3  # V, X, /

# 모델 설계
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 전처리 및 증강
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# 모델 학습
model.fit(train_generator, epochs=1000)

# 모델 저장
model.save('image_grading_model.h5')
