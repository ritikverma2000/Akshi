import os
import pyttsx3
from flask import Flask, render_template, Response, request
import cv2
from keras.preprocessing import image
import numpy as np
from keras.models import model_from_json

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


def text_to_speech(text):
    engine = pyttsx3.init()

    engine.say(text)
    engine.runAndWait()


def gen_frames():
    json_file = open('ModelFiles//fer.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # # Load weights and them to model

    model.load_weights('ModelFiles//fer.h5')
    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 6, minSize=(150, 150))
            for (x, y, w, h) in faces_detected:
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
                roi_gray = gray_img[y:y + w, x:x + h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255.0

                predictions = model.predict(img_pixels)
                max_index = int(np.argmax(predictions))
                emotions = ['angry', 'happy', 'neutral', 'sad']
                predicted_emotion = emotions[max_index]
                text_to_speech(predicted_emotion)

                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                            2)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
