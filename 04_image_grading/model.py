import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw


def build_model(input_shape=(64, 64, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, train_data_dir, validation_data_dir):
    train_datagen = image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_data_dir, target_size=(64, 64), batch_size=32,
                                                     class_mode='categorical')
    validation_set = test_datagen.flow_from_directory(validation_data_dir, target_size=(64, 64), batch_size=32,
                                                      class_mode='categorical')

    model.fit(training_set, epochs=10, validation_data=validation_set)
    model.save('image_grading_model.h5')


def load_model():
    return tf.keras.models.load_model('image_grading_model.h5')


def predict_marks(model, image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_map = {0: 'V', 1: 'X', 2: '/'}
    grading_info = {}

    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    score = 100

    for i, pred in enumerate(predictions[0]):
        label = class_map[i]
        count = int(pred * 10)  # 단순 예시로 각 레이블의 빈도 추정
        color_map = {"V": "blue", "X": "red", "/": "yellow"}
        grading_info[label] = (count, color_map[label])

        if label == 'V':
            score -= 10 * count
        elif label == 'X':
            score -= 5 * count
        elif label == '/':
            score -= 1 * count

        for _ in range(count):
            draw.text((10 + i * 30, 10), label, fill=color_map[label])

    graded_filename = f'graded_{os.path.basename(image_path)}'
    graded_path = os.path.join('static/graded', graded_filename)
    img.save(graded_path)

    return score, grading_info, graded_filename
