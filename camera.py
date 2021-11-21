import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import winsound

# new_model = tf.keras.models.load_model('Model.h5')
#
# frequency = 2500
# duration = 1000
# face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# eye = cv2.CascadeClassifier('haarcascade_eye.xml')
#
class Video(object):
    counter = 0
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        new_model = tf.keras.models.load_model('Model.h5')
        frequency = 2500
        duration = 1000
        face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        eye = cv2.CascadeClassifier('haarcascade_eye.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye.detectMultiScale(gray, 1.1, 4)
        # faces = face.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in eyes:
            # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            eyess = eye.detectMultiScale(roi_gray)

            if len(eyess) == 0:
                print("eyes are not detected")
                eyes_roi=np.array([0])

            else:
                for (ex, ey, ew, eh) in eyess:
                    eyes_roi = roi_color[ey:ey + eh, ex:ex + ew]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(face.empty())

        faces = face.detectMultiScale(gray, 1.1, 4)
        # faces=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMG)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        final_img = cv2.resize(eyes_roi, (224, 224), cv2.INTER_AREA)
        final_img = np.expand_dims(final_img, axis=0)  # need fourth dimension
        final_img = final_img / 255.0  # normalize

        Predictions = new_model.predict(final_img)

        if Predictions > 0.5:
            status = "ACTIVE"
            cv2.putText(frame, status, (150, 150), font, 1, (0, 255, 0), 2, cv2.LINE_4)
            x1, y1, w1, h1 = 0, 0, 175, 75

            # draw black rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (255, 255, 255), -1)
            # add text
            cv2.putText(frame, 'Eyes open', (x1 + int(w1 / 10), y1 + int(h1 / 2)), font, 0.7, (0, 255, 0), 2)

        else:
            counter = counter + 1
            status = "DROWSY"
            cv2.putText(frame, status, (150, 150), font, 1, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if counter > 5:
                x1, y1, w1, h1 = 0, 0, 175, 75
                # draw black rectangle
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                # add text
                cv2.putText(frame, 'SLEEP ALERT!!!', (x1 + int(w1 / 10), y1 + int(h1 / 2)), font, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
                counter = 0
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
