import time
import threading
import pyttsx3
from flask import Flask, render_template, Response, request
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json
import os
from gtts import gTTS
import playsound
from threading import Event
import tensorflow as tf

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


check_happy = False
check_sad = False
check_angry = False
check_neutral = False


def gen_frames():
    emotion_interpreter = tf.lite.Interpreter(
        model_path="C://Users//varma//Desktop//MajorProjectReportFiles//ModelFiles//emotion_detection_model_60epochs"
                   ".tflite")
    emotion_interpreter.allocate_tensors()
    emotion_input_details = emotion_interpreter.get_input_details()
    emotion_output_details = emotion_interpreter.get_output_details()
    cap = cv2.VideoCapture(0)
    timemark = time.time()
    fpsFilter = 0

    while True:

        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            face_classifier = cv2.CascadeClassifier('E://MajorProjectLatest//test_fol'
                                                    '//haarcascade_frontalface_default.xml')
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            class_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

            # ret, frame = cap.read()
            # labels = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                x1, y1 = x + w, y + h

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

                cv2.line(frame, (x, y), (x + 30, y), (0, 0, 255), 6)  # Top Left
                cv2.line(frame, (x, y), (x, y + 30), (0, 0, 255), 6)

                cv2.line(frame, (x1, y), (x1 - 30, y), (0, 0, 255), 6)  # Top Right
                cv2.line(frame, (x1, y), (x1, y + 30), (0, 0, 255), 6)

                cv2.line(frame, (x, y1), (x + 30, y1), (0, 0, 255), 6)  # Bottom Left
                cv2.line(frame, (x, y1), (x, y1 - 30), (0, 0, 255), 6)

                cv2.line(frame, (x1, y1), (x1 - 30, y1), (0, 0, 255), 6)  # Bottom right
                cv2.line(frame, (x1, y1), (x1, y1 - 30), (0, 0, 255), 6)

                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                # Get image ready for prediction
                roi = roi_gray.astype('float') / 255.0  # Scale
                roi = image.img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

                emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
                emotion_interpreter.invoke()
                emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

                # preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 4 classes
                emotion_label = class_labels[emotion_preds.argmax()]  # Find the label
                emotion_label_position = (x, y)

                dt = time.time() - timemark
                timemark = time.time()
                fps = 1 / dt
                fpsFilter = .95 * fpsFilter + .05 * fps

                cv2.putText(frame, str(round(fpsFilter, 1)) + '  fps  ' + emotion_label,
                            emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)

                if emotion_label == 'Happy' or emotion_label == 'Sad' or emotion_label == 'Angry' or emotion_label == 'Neutral':
                    global check_happy
                    global check_sad
                    global check_angry
                    global check_neutral
                    if emotion_label == 'Happy' and check_happy == False:
                        filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
                                   "//MajorProject//AudioFiles//happy.mp3"

                        check_happy = True
                        playsound.playsound(filename)
                        check_angry = False
                        check_neutral = False
                        check_sad = False

                    if emotion_label == 'Sad' and check_sad == False:
                        filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
                                   "//MajorProject//AudioFiles//sad.mp3"
                        check_sad = True
                        playsound.playsound(filename)
                        check_happy = False
                        check_angry = False
                        check_neutral = False
                    if emotion_label == 'Angry' and check_angry == False:
                        filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
                                   "//MajorProject//AudioFiles//angry.mp3"
                        check_angry = True
                        playsound.playsound(filename)
                        check_happy = False
                        check_sad = False
                        check_neutral = False
                    if emotion_label == 'Neutral' and check_neutral == False:
                        filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
                                   "//MajorProject//AudioFiles//neutral.mp3"
                        check_neutral = True
                        playsound.playsound(filename)
                        check_happy = False
                        check_angry = False
                        check_sad = False

            #
            #
            #
            #
            #
            #     # playsound.playsound(filename)
            #
            #
            # if emotion_label == 'Sad':
            #     filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
            #                "//MajorProject//AudioFiles//sad.mp3"
            #
            #     playsound.playsound(filename)
            #
            #     playsound.playsound(filename)
            # if emotion_label == 'Angry':
            #     filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
            #                "//MajorProject//AudioFiles//angry.mp3"
            #
            #     playsound.playsound(filename)
            #
            # if emotion_label == 'Neutral':
            #     filename = "C://Users//varma//Desktop//MajorProjectReportFiles//CODE_FILES//WORK" \
            #                "//MajorProject//AudioFiles//neutral.mp3"
            #
            #     playsound.playsound(filename)

        ret, buffer = cv2.imencode('.jpg', cv2.resize(frame, (1000, 700)))
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/testcamera")
def camera():
    return render_template('testcamera.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
