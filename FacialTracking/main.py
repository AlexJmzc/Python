# Código que utiliza un modelo entrenado en cascada para hacer un tracking de rostros en cámara
# Code that uses a cascade trained model for face tracking in real time 

# Instalar cv2 con el comando -> pip install cv2
# Install cv2 with the command -> pip install cv2

# Para cerrar la ventana de la cámara se presiona la tecla Esc
# For closing the camera window press Esc 

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imshow('img', img)
       
    k = cv2.waitKey(30)
    if k == 27:
        break
cap.release()