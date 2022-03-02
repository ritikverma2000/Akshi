import os
import datetime, time
from threading import Thread

from flask import Flask, render_template, Response, request
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

import emotion

app = Flask(__name__)

global capture, rec_frame, grey, switch, ans, camera

capture = 0

switch = 1

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass


def gen_frames():
    # generate frame by frame from camera
    global out, capture, rec_frame, ans, resized_img
    json_file = open('E://MajorProjectLatest//test_fol//fer.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # # Load weights and them to model
    # """ model.load_weights('E:/MajorProjectLatest/fer.h5')  """
    model.load_weights('E://MajorProjectLatest//test_fol//fer.h5')

    while True:
        success, frame = camera.read()

        if success:

            face_haar_cascade = cv2.CascadeClassifier('E://MajorProjectLatest//test_fol'
                                                      '//haarcascade_frontalface_default.xml')

            ret, img = camera.read()
            if not ret:
                break

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))
            for (x, y, w, h) in faces_detected:
                x1, y1 = x + w, y + h
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.line(img, (x, y), (x + 30, y), (0, 0, 255), 6)  # Top Left
                cv2.line(img, (x, y), (x, y + 30), (0, 0, 255), 6)

                cv2.line(img, (x1, y), (x1 - 30, y), (0, 0, 255), 6)  # Top Right
                cv2.line(img, (x1, y), (x1, y + 30), (0, 0, 255), 6)

                cv2.line(img, (x, y1), (x + 30, y1), (0, 0, 255), 6)  # Bottom Left
                cv2.line(img, (x, y1), (x, y1 - 30), (0, 0, 255), 6)

                cv2.line(img, (x1, y1), (x1 - 30, y1), (0, 0, 255), 6)  # Bottom right
                cv2.line(img, (x1, y1), (x1, y1 - 30), (0, 0, 255), 6)
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0

                predictions = model.predict(img_pixels)
                max_index = int(np.argmax(predictions))
                emotions = ['angry', 'happy', 'neutral', 'sad']
                predicted_emotion = emotions[max_index]
                emotion.x = predicted_emotion

                cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                            2)

            ret, buffer = cv2.imencode('.jpg', cv2.resize(img, (1000, 700)))
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        try:

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

    else:
        pass


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/camera")
def camera():
    return render_template('camera.html')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, loaded_model_json, model, json_file
    if request.method == 'POST':

        if request.form.get('stop') == 'Stop/Start':

            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:

                camera = cv2.VideoCapture(0)

                switch = 1



    elif request.method == 'GET':
        return render_template('camera.html')
    return render_template('camera.html',data=emotion.x)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
