import cv2

# It is an Object Detection Algorithm used to identify faces in an image or a real time video.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create variable to store video using VideoCapture() function. Pass parameter 0 in VideoCapture(0) to access webcam

video = cv2.VideoCapture(0)
# counting number of frames
a = 1

while True:
    # incrementing the number of frames till the condition is true
    a = a + 1
    check, frame = video.read()
    print(frame)

    # we read the image and convert it to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow('Face Detector', frame)

    # new frame after every one second
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# prints number of frames
print(a)
# release() -> Closes video file or capturing device.
video.release()
cv2.destroyAllWindows()
