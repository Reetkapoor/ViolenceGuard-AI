import os
import cv2
import numpy as np
import random
from collections import deque
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Input, Dense, Dropout, LSTM, Bidirectional,
                          TimeDistributed, Flatten)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import to_categorical, plot_model

# -------------------- CONFIG --------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 96, 96
SEQUENCE_LENGTH = 16
DATASET_DIR = "Training & Evaluation/training_data"  # adjust if mounted differently
CLASSES_LIST = ["NonViolence", "Violence"]
EPOCHS = 20
BATCH_SIZE = 8
MODEL_SAVE_PATH = "Weights/violence_detector.keras"

# -------------------- DATASET --------------------
def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = (resized_frame / 255.0).astype(np.float32)
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def create_dataset():
    features, labels = [], []
    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f"Extracting {class_name}...")
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        for file_name in files_list:
            video_path = os.path.join(DATASET_DIR, class_name, file_name)
            frames = frames_extraction(video_path)
            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)

    return np.asarray(features, dtype=np.float32), np.array(labels)

# -------------------- MODEL --------------------
def create_model():
    mobilenet = MobileNetV2(include_top=False, weights="imagenet")
    mobilenet.trainable = True
    for layer in mobilenet.layers[:-60]:
        layer.trainable = False

    model = Sequential()
    model.add(Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(mobilenet))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(len(CLASSES_LIST), activation="softmax"))
    return model

# -------------------- TRAINING --------------------
if __name__ == "__main__":
    features, labels = create_dataset()
    one_hot_labels = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        features, one_hot_labels, test_size=0.1, random_state=42, shuffle=True
    )

    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.6, patience=5,
                                  min_lr=5e-5, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr]
    )

    print("Evaluating on test set...")
    model.evaluate(X_test, y_test)

    os.makedirs("Weights", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")
    model.save("/Weights/violence_detector_v1.h5")
    print("Also saved as .h5")
