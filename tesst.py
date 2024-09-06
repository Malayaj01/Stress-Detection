import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model('model_file.keras')

# Initialize the webcam
video = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict = {0:'angry', 1:'disgust', 2:'fearful', 3:'happy', 4:'neutral', 5:'sad', 6:'surprised'}



if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img, (48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame, (x,y),(x+w,y+h), (50,50,255),2)
        cv2.rectangle(frame, (x,y-40), (x+w,y), (50,50,255), -1)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    
    

# Release the webcam and close windows
video.release()
cv2.destroyAllWindows()
