import os
import cv2

models_dir = './models'

face_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()

    if not _:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(frame_gray, minNeighbors=25)

    for face in faces:
        x, y, w, h = face

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):                
        break

video.release()
cv2.destroyAllWindows()