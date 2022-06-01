import os
import threading
import argparse
from time import time
import tkinter as tk
from tkinter import filedialog

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import cv2
import numpy as np
import dlib
import visualkeras
from pynput.keyboard import Key, Controller

# global configs
keyboard = Controller()
tf.get_logger().setLevel('ERROR')
root = tk.Tk()
root.withdraw()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='Train model')
parser.add_argument('--test', action='store_true', help='Test model')
parser.add_argument('--unlimited', action='store_true', help='Unlimited mode for input')
parser.add_argument('--continue-training', action='store_true', help='Continue training')
parser.add_argument('--method', type=str, help='Method to use in eye detection')
parser.add_argument('--mode', type=str, help='Mode to run in')
parser.add_argument('--plot', action='store_true', help='plot model loss after training')
parser.add_argument('--save-best', action='store_true', help='plot model loss after training')
parser.add_argument('--eye-mode', type=str, help='Mode to handle eye states')
args = parser.parse_args()


# global variables
IMG_SIZE = (32, 32)
EYE_BORDER_COLOR = (0, 0, 255)
TEST_RATIO = .2
PADDING = 15
EPOCHS = 100

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# file paths
DATASET_PATH = './data'
CHECKPOINTS_PATH = './checkpoints'
DATASET_NAME = 'mrlEyes_2018_01'
CURR_CHECKPOINT_NAME = 'training_7'

CURR_DATASET_PATH = f"{DATASET_PATH}/{DATASET_NAME}"
CHECKPOINT_PATH = f"{CHECKPOINTS_PATH}/{CURR_CHECKPOINT_NAME}/cp.ckpt"

USE_SELF_COLLECTED_DATA_ONLY = True
SUBJECTS = ["s0038", "s0039", "s0040"] # [Emir, Samet, Kübra]
CURR_SUBJECT = SUBJECTS[0]

# dataset headers
HEADERS = [
    'subject_id', # person (can be dropped)
    'image_id',
    'gender', # 0 (man) - 1 (woman)
    'glasses',  # 0 (no) - 1 (yes)
    'eye_state',  # 0 (closed) - 1 (open)
    'reflections',  # 0 (none) - 1 (small) - 2 (big)
    'lightning_conditions / image_quality', # 0 (bad) - 1 (good)
    'sensor_id', # 01 (RealSense) - 02 (IDS) - 03 (Aptina)
]

def strip_extension(filename) -> str:
    """Returns file name without extension"""
    return os.path.splitext(filename)[0]

def parse_mrl_eyes_filename(filename:str, header:str) -> str:
    """Returns the label for the given data."""
    stripped_filename = strip_extension(filename)
    features = stripped_filename.split('_')
    return features[HEADERS.index(header)]

def load_mrl_eyes_dataset(path:int, size:int=None) -> list:
    """Loads the data from the given path."""
    dataset = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if USE_SELF_COLLECTED_DATA_ONLY and dir not in SUBJECTS:
                continue
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    img = os.path.join(root, dir, file)
                    img = cv2.imread(img)
                    img = cv2.resize(img, IMG_SIZE)
                    label = parse_mrl_eyes_filename(file, 'eye_state')
                    dataset.append([img, [int(label)]])

                    if size and len(dataset) >= size:
                        return dataset
    return dataset

def load_dataset() -> tuple:
    """load and shape the dataset"""
    dataset = load_mrl_eyes_dataset(CURR_DATASET_PATH)
    np.random.shuffle(dataset)
    X, y = [img for img, _ in dataset], [label for _, label in dataset]
    X = np.array(X).reshape(-1, *IMG_SIZE, 3)
    y = np.array(y)

    split_index = int(len(X) * (1 - TEST_RATIO))
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]

def create_model() -> keras.Model:
    """Create a CNN model"""
    model = keras.models.Sequential()

    # data augmentation
    model.add(layers.experimental.preprocessing.RandomRotation(0.1))
    model.add(layers.experimental.preprocessing.RandomContrast(0.1))

    # convolutional layers
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    # fully connected layers
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))
    model.compile(
        optimizer = 'adam', 
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )

    visualize_model(model, to_file='model_architecture.png')
    return model

def visualize_model(model, to_file):
    try:
        visualkeras.layered_view(
            model,
            to_file=to_file,
            legend=True,
            type_ignore=[
                layers.experimental.preprocessing.RandomFlip,
                layers.experimental.preprocessing.RandomRotation,
                layers.experimental.preprocessing.RandomContrast,
                layers.experimental.preprocessing.RandomTranslation,
                layers.experimental.preprocessing.RandomZoom
            ],
            scale_xy = 10,
            scale_z = 0.1
        )
    except AttributeError:
        print("Visualization is not supported for this structure. Try removing experimental layers")

def plot_history(history:keras.callbacks.History) -> None:
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    print("final val loss:", history.history['val_loss'][-1])
    print("best val loss:", min(history.history['val_loss']))
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    print("final val accuracy:", history.history['val_accuracy'][-1])
    print("best val accuracy:", max(history.history['val_accuracy']))
    plt.show()

def train_model(model, X_train, y_train, X_test, y_test):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                    save_weights_only=True,
                                                    save_best_only=args.save_best,
                                                    verbose=1)

    history = model.fit(X_train, y_train, epochs=EPOCHS, 
              validation_data=(X_test, y_test),
              callbacks=[cp_callback])

    if args.plot:
        plot_history(history)

    return model

def split_eyes(eyes):
    """split eyes by coordinates"""
    eyes_left = eyes[0]
    eyes_right = eyes[1]
    if eyes_left[0] > eyes_right[0]:
        eyes_left, eyes_right = eyes_right, eyes_left
    return eyes_left, eyes_right

def normalize(x:list):
    """Normalize pixel values"""
    return x / 255.0

def handle_eye_conditions(leye_cond, reye_cond, mode="keyboard"):
    """handles eye conditions, mode is one of 'keyboard', 'volume'"""
    if mode == 'keyboard':
        if leye_cond == 0 and reye_cond == 0:
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
        elif leye_cond == 0:
            keyboard.press(Key.down)
            keyboard.release(Key.down)
        elif reye_cond == 0:
            keyboard.press(Key.up)
            keyboard.release(Key.up)
    elif mode == 'volume':
        if leye_cond == 0 and reye_cond == 0:
            keyboard.press(Key.media_volume_mute)
            keyboard.release(Key.media_volume_mute)
        elif leye_cond == 0:
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
        elif reye_cond == 0:
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
    else:
        raise ValueError("Invalid mode")

def get_eyes_dlib(frame):

    # Convert to dlib
    frame_temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # dlib face detection
    detector = dlib.get_frontal_face_detector()
    detections = detector(frame_temp, 1)

    # Find landmarks
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = dlib.full_object_detections()
    for det in detections:
        faces.append(sp(frame_temp, det))

    # Bounding box and eyes
    bb = [i.rect for i in faces]
    bb = [((i.left(), i.top()),
        (i.right(), i.bottom())) for i in bb]                            # Convert out of dlib format
    # Display
    for i in bb:
        cv2.rectangle(frame, i[0], i[1], (255, 0, 0), 5)     # Bounding box

    right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
    right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]          # Convert out of dlib format

    left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
    left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]            # Convert out of dlib format

    return left_eyes, right_eyes

def get_eye_prediction_dlib(eye, frame, model, padding=40):
    cv2.rectangle(frame, (max(eye, key=lambda x: x[0])[0]+padding, max(eye, key=lambda x: x[1])[1]+padding),
                        (min(eye, key=lambda x: x[0])[0]-padding, min(eye, key=lambda x: x[1])[1]-padding),
                        (0, 0, 255), 5)
    for point in eye:
        cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)

    eye_x = min(eye, key=lambda x: x[0])[0] -padding
    eye_y = min(eye, key=lambda x: x[1])[1] -padding
    eye_w = max(eye, key=lambda x: x[0])[0] +padding - eye_x
    eye_h = max(eye, key=lambda x: x[1])[1] +padding - eye_y
    eye_gray = cv2.cvtColor(frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w], cv2.COLOR_BGR2GRAY)

    eye_gray_rgb = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)
    eye_gray_resized = cv2.resize(eye_gray_rgb, IMG_SIZE)
    eye_gray_reshaped = eye_gray_resized.reshape(-1, *IMG_SIZE, 3)

    eye_pred = model.predict(eye_gray_reshaped)
    eye_cond = np.argmax(eye_pred)
    eye_prob = np.amax(eye_pred)*100
    eye_state = 'open' if eye_cond == 1 else 'closed'
    return eye_cond, eye_prob, eye_state

def handle_camera_mode_dlib(model):
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        left_eyes, right_eyes = get_eyes_dlib(frame)

        for leye, reye in zip(left_eyes, right_eyes):
            leye_cond, leye_prob, leye_state = get_eye_prediction_dlib(
                eye=leye,
                frame=frame,
                model=model,
                padding=PADDING
            )
            reye_cond, reye_prob, reye_state = get_eye_prediction_dlib(
                eye=reye,
                frame=frame,
                model=model,
                padding=PADDING
            )

            cv2.putText(frame, f'L eye: {leye_state} (%{leye_prob}) {confidence(leye_prob)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
            cv2.putText(frame, f'R eye: {reye_state} (%{reye_prob}) {confidence(reye_prob)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
            if leye_cond == reye_cond:
                both_state = 'open' if leye_cond == 1 else 'closed'
                cv2.putText(frame, f'both eyes are {both_state}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def handle_input_mode_dlib(model):
    while True:
        frame = cv2.imread(filedialog.askopenfilename())

        left_eyes, right_eyes = get_eyes_dlib(frame)

        for leye, reye in zip(left_eyes, right_eyes):
            leye_cond, leye_prob, leye_state = get_eye_prediction_dlib(
                eye=leye,
                frame=frame,
                model=model,
                padding=PADDING
            )
            reye_cond, reye_prob, reye_state = get_eye_prediction_dlib(
                eye=reye,
                frame=frame,
                model=model,
                padding=PADDING
            )

            cv2.putText(frame, f'L eye: {leye_state} (%{leye_prob}) {confidence(leye_prob)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
            cv2.putText(frame, f'R eye: {reye_state} (%{reye_prob}) {confidence(reye_prob)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
            if leye_cond == reye_cond:
                both_state = 'open' if leye_cond == 1 else 'closed'
                cv2.putText(frame, f'both eyes are {both_state}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)

            cv2.imshow("output", frame)
            cv2.waitKey(0)

        if not args.unlimited:
            break

def handle_input_mode_haar(model):
    while True:
        frame = cv2.imread(filedialog.askopenfilename())

        grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_scale_rgb = cv2.cvtColor(grey_scale, cv2.COLOR_GRAY2BGR)

        faces = face_cascade.detectMultiScale(grey_scale_rgb, 1.3, 5)
        roi_gray = None
        leye_gray = None
        reye_gray = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = grey_scale_rgb[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5)
            if len(eyes) == 2:
                left_eye, right_eye = split_eyes(eyes)
                lex, ley, lew, leh = left_eye
                rex, rey, rew, reh = right_eye

                cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), EYE_BORDER_COLOR, 5)
                cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), EYE_BORDER_COLOR, 5)

                leye_gray = roi_gray[ley:ley+leh, lex:lex+lew]
                leye_gray = cv2.resize(leye_gray, IMG_SIZE)
                leye_gray_reshaped = leye_gray.reshape(-1, *IMG_SIZE, 3)

                reye_gray = roi_gray[rey:rey+reh, rex:rex+rew]
                reye_gray = cv2.resize(reye_gray, IMG_SIZE)
                reye_gray_reshaped = reye_gray.reshape(-1, *IMG_SIZE, 3)

                leye_pred = model.predict(leye_gray_reshaped)
                reye_pred = model.predict(reye_gray_reshaped)

                leye_cond = np.argmax(leye_pred)
                reye_cond = np.argmax(reye_pred)
                
                leye_prob = np.amax(leye_pred)*100
                reye_prob = np.amax(reye_pred)*100

                leye_state = 'open' if leye_cond == 1 else 'closed'
                reye_state = 'open' if reye_cond == 1 else 'closed'

                cv2.putText(frame, f'L eye: {leye_state} (%{leye_prob}) {confidence(leye_prob)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
                cv2.putText(frame, f'R eye: {reye_state} (%{reye_prob}) {confidence(reye_prob)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
                if leye_cond == reye_cond:
                    both_state = 'open' if leye_cond == 1 else 'closed'
                    cv2.putText(frame, f'both eyes are {both_state}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)

                cv2.putText(roi_color, leye_state, (lex, ley - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_BORDER_COLOR, 2)
                cv2.putText(roi_color, reye_state, (rex, rey - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_BORDER_COLOR, 2)

        cv2.imshow('frame', frame)
        cv2.waitKey(0)

        if not args.unlimited:
            break

def handle_camera_mode_haar(model):
    cap = cv2.VideoCapture(0)
    cap_mode = 1
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        grey_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey_scale_bgr = cv2.cvtColor(grey_scale, cv2.COLOR_GRAY2BGR)

        faces = face_cascade.detectMultiScale(grey_scale_bgr, 1.3, 5)
        roi_gray = None
        leye_gray = None
        reye_gray = None
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = grey_scale_bgr[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5)
            if len(eyes) == 2:
                left_eye, right_eye = split_eyes(eyes)
                lex, ley, lew, leh = left_eye
                rex, rey, rew, reh = right_eye

                cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), EYE_BORDER_COLOR, 5)
                cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), EYE_BORDER_COLOR, 5)

                leye_gray = roi_gray[ley:ley+leh, lex:lex+lew]
                leye_gray = cv2.resize(leye_gray, IMG_SIZE)
                leye_gray_reshaped = leye_gray.reshape(-1, *IMG_SIZE, 3)

                reye_gray = roi_gray[rey:rey+reh, rex:rex+rew]
                reye_gray = cv2.resize(reye_gray, IMG_SIZE)
                reye_gray_reshaped = reye_gray.reshape(-1, *IMG_SIZE, 3)

                leye_pred = model.predict(leye_gray_reshaped)
                reye_pred = model.predict(reye_gray_reshaped)

                leye_cond = np.argmax(leye_pred)
                reye_cond = np.argmax(reye_pred)
                
                leye_prob = np.amax(leye_pred)*100
                reye_prob = np.amax(reye_pred)*100

                leye_state = 'open' if leye_cond == 1 else 'closed'
                reye_state = 'open' if reye_cond == 1 else 'closed'

                eye_mode = 'keyboard' if not args.eye_mode else args.eye_mode
                handle_eye_conditions(leye_cond, reye_cond, mode=eye_mode)

                cv2.putText(frame, f'L eye: {leye_state} (%{leye_prob}) {confidence(leye_prob)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
                cv2.putText(frame, f'R eye: {reye_state} (%{reye_prob}) {confidence(reye_prob)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)
                if leye_cond == reye_cond:
                    both_state = 'open' if leye_cond == 1 else 'closed'
                    cv2.putText(frame, f'both eyes are {both_state}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, EYE_BORDER_COLOR, 2)


                cv2.putText(roi_color, '%'+str(leye_prob), (lex, ley - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EYE_BORDER_COLOR, 2)
                cv2.putText(roi_color, '%'+str(reye_prob), (rex, rey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, EYE_BORDER_COLOR, 2)

                cv2.putText(roi_color, leye_state, (lex, ley - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_BORDER_COLOR, 2)
                cv2.putText(roi_color, reye_state, (rex, rey - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_BORDER_COLOR, 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('m'):
            cap_mode = int(not cap_mode)
            print("Changed mode to", "open eyes" if cap_mode else "closed eyes")
        elif key == ord('s'):
            cv2.imwrite(str(time())+'.png', frame)
            print("saved")
        elif key == ord('q'):
            break
        elif ord('"') < key < ord('z'):
            if roi_gray is not None:
                if key == ord('r'):
                    if reye_gray is not None:
                        cv2.imwrite(f'{CURR_DATASET_PATH}/{CURR_SUBJECT}/{CURR_SUBJECT}_{int(time())}_0_0_{cap_mode}_0_0_01.png', reye_gray)
                        print("saved right eye")
                    else:
                        print("right eye not found")
                if key == ord('l'):
                    if leye_gray is not None:
                        cv2.imwrite(f'{CURR_DATASET_PATH}/{CURR_SUBJECT}/{CURR_SUBJECT}_{int(time())}_0_0_{cap_mode}_0_0_01.png', leye_gray)
                        print("saved left eye")
                    else:
                        print("left eye not found")
                elif key == ord('b'):
                    if reye_gray is not None and leye_gray is not None:
                        cv2.imwrite(f'{CURR_DATASET_PATH}/{CURR_SUBJECT}/{CURR_SUBJECT}_{int(time())}_0_0_{cap_mode}_0_0_01.png', reye_gray)
                        cv2.imwrite(f'{CURR_DATASET_PATH}/{CURR_SUBJECT}/{CURR_SUBJECT}_{int(time())+1}_0_0_{cap_mode}_0_0_01.png', leye_gray)
                        print("saved both eyes")
                    else:
                        print("both eyes not found")

    cap.release()
    cv2.destroyAllWindows()

def confidence(prob):
    if prob < 40:
        return 'low confidence'
    elif prob < 75:
        return 'medium confidence'
    return 'high confidence'

SELECTED_COLOR = 'red'
NAVIGATE_COLOR = 'blue'
UNSELECTED_COLOR = 'white'
buttons = []
current_row = 0
selected_row = -1
def handle_gui(model):
    import keyboard

    def update_colors(navigated_row:int, selected_row:int):
        for i, button in enumerate(buttons):
            if i == navigated_row:
                button['bg'] = NAVIGATE_COLOR
            if i == selected_row:
                button['bg'] = SELECTED_COLOR
            if i != navigated_row and i != selected_row:
                button['bg'] = UNSELECTED_COLOR

    def navigate(row, col):
        global current_row
        current_row = row
        update_colors(row, selected_row)

    def select(row, col):
        global selected_row
        selected_row = row
        label.configure(text="You've selected option "+str(row+1))

    root = tk.Tk()
    for row in range(0,7):
        button = tk.Button(root, text="Option "+str(row+1), 
                            command=lambda row=row, col=0: navigate(row, col))
        button.grid(row=row, column=0, sticky="nsew")
        buttons.append(button)

    label = tk.Label(root, text="")
    label.grid(row=len(buttons), column=0, sticky="new")

    root.grid_rowconfigure(10, weight=1)

    def on_down_arrow_press(e):
        global current_row
        current_row = current_row + 1 if current_row < len(buttons) - 1 else len(buttons) - 1
        navigate(current_row, 0)

    def on_up_arrow_press(e):
        global current_row
        current_row = current_row - 1 if current_row > 0 else 0
        navigate(current_row, 0)

    def on_enter_press(e):
        select(current_row, 0)
        update_colors(current_row, selected_row)

    keyboard.on_press_key("up arrow", on_up_arrow_press)
    keyboard.on_press_key("down arrow", on_down_arrow_press)
    keyboard.on_press_key("enter", on_enter_press)

    navigate(current_row, 0)
    on_enter_press(None)

    threading.Thread(target=handle_camera_mode_haar, args=(model,)).start()
    root.mainloop()



def main():
    model = create_model()

    if args.train or args.test:
        X_train, y_train, X_test, y_test = load_dataset()

        X_train, X_test = normalize(X_train), normalize(X_test)

        if args.train:
            if args.continue_training:
                model.load_weights(CHECKPOINT_PATH)
            model = train_model(model, X_train, y_train, X_test, y_test)

        if args.test:
            model.load_weights(CHECKPOINT_PATH)
            model.evaluate(X_test, y_test, verbose=2)
    else:
        model.load_weights(CHECKPOINT_PATH) 

    if args.mode == 'camera':
        if args.method == 'dlib':
            handle_camera_mode_dlib(model)
        elif args.method == 'haar':
            handle_camera_mode_haar(model)
        else:
            handle_camera_mode_haar(model)
    elif args.mode == 'input':
        if args.method == 'dlib':
            handle_input_mode_dlib(model)
        elif args.method == 'haar':
            handle_input_mode_haar(model)
        else:
            handle_input_mode_dlib(model)
    elif args.mode == 'gui':
        handle_gui(model)
    elif not args.mode:
        return
    else:
        raise ValueError("Invalid mode name")


if __name__ == "__main__": main()
