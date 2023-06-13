# Código que escanea una imagen y extrae el texto que contiene
# Code that scans an image and extract the text that contains on it

# Instalar cv2 con el comando -> pip install cv2
# Install cv2 with the command -> pip install cv2

# Instalar pytesseract con el comando -> pip install pytesseract
# Install pytesseract with the commando -> pip install pytesseract

# Descargar el instalador de tesseract para Windows desde: https://github.com/UB-Mannheim/tesseract/wiki
# Download the tesseract installer for Windows from: https://github.com/UB-Mannheim/tesseract/wiki

# Descargar lenguajes de tesseract desde: https://github.com/tesseract-ocr/tessdata_best
# Download languajes of tesseract from: https://github.com/tesseract-ocr/tessdata_best

# Con el lenguaje descargado, ponerlo en la carpeta "tessdata"
# With the languaje downloaded, put it on the "tessdata" directory

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

image = cv2.imread('text1.jpeg')
#image = cv2.resize(image, (550,550))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 10, 150)
canny = cv2.dilate(canny, None, iterations=1)

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

for c in cnts:
    epsilon = 0.01*cv2.arcLength(c,True)
    aprox = cv2.approxPolyDP(c, epsilon, True)
    
    if len(aprox) >= 4:
        cv2.drawContours(image, [aprox], 0, (0,255,255), 2)

        cv2.circle(image, tuple(aprox[1][0]), 7, (255,0,0), 2)
        cv2.circle(image, tuple(aprox[0][0]), 7, (9,255,0), 2)
        cv2.circle(image, tuple(aprox[2][0]), 7, (0,0,255), 2)
        cv2.circle(image, tuple(aprox[3][0]), 7, (255,255,0), 2)

    text = pytesseract.image_to_string(canny, lang='spa')
    print('Text: ', text)

cv2.imshow('Img', image)
cv2.imshow('Text', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()