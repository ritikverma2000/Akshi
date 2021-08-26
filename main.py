

import cv2



face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
a = 1


while True:
    a = a + 1
    check , frame = video.read()
    print(frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1) # new frame after every one second

    if key == ord('q'):
     break
print(a) # prints number of frames
video.release()
cv2.destroyAllWindows()

