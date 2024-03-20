import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.utils import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
# load model
model = load_model("best_model_4.h5")



cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    img_rgb=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    faces_detected = face_recognition.face_locations(img_rgb)


    for loc in faces_detected:
        cv2.rectangle(test_img, (loc[3], loc[0]), (loc[1], loc[2]), (0, 255, 0), thickness=7) 
        roi_gray = gray_img[loc[0]:loc[2], loc[3]:loc[1]]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotions=('angry','happy','neutral','sad')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion,( int(loc[3]), int(loc[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows