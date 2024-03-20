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


img=cv2.imread('test_images/test2.jpg')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

face_loc=face_recognition.face_locations(img_rgb)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

for loc in face_loc:
    cv2.rectangle(img, (loc[3], loc[0]), (loc[1], loc[2]), (0, 255, 0), thickness=2) 
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

    cv2.putText(img, predicted_emotion,( int(loc[3]), int(loc[0])), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

resized_img = cv2.resize(img, (1000, 700))
cv2.imshow('Facial emotion analysis ', resized_img)

cv2.waitKey(0)


cv2.destroyAllWindows