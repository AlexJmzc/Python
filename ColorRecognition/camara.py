# Detección de colores en tiempo real con un modelo entrenado de KNN
# Colour detection in real time with a trained KNN model

import math
import pickle
import statistics
import time

import cv2
import numpy as np
from sklearn import preprocessing

import tkinter as tk
from tkinter import messagebox
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

root = tk.Tk()
root.title("COLORES")

label = tk.Label(root, text="Colores")
label.pack()


#METODO PARA CALCULAR LAS MEDIAS EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE MEAN IN EACH SPECTRUM WITH A 5 "WINDOW"
def media(vector):
    longitud = len(vector)
    mediaC = []
    for i in range(1,longitud,5):
        mean = statistics.mean(vector[i:i+4])
        mediaC.append(mean)

    return mediaC

#METODO PARA CALCULAR LAS MODAS EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE MODE IN EACH SPECTRUM WITH A 5 "WINDOW"
def moda(vector):
    longitud = len(vector)
    modaC = []
    for i in range(1,longitud,5):
        mode = statistics.mode(vector[i:i+4])
        modaC.append(mode)

    return modaC

#METODO PARA CALCULAR LAS MEDIANAS EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE MEDIAN IN EACH SPECTRUM WITH A 5 "WINDOW"
def mediana(vector):
    longitud = len(vector)
    medianaC = []
    for i in range(1,longitud,5):
        median = statistics.median(vector[i:i+4])
        medianaC.append(median)

    return medianaC

#METODO PARA CALCULAR LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE ESTANDAR DEVIATION IN EACH SPECTRUM WITH A 5 "WINDOW"
def desvEstandar(vector):
    longitud = len(vector)
    desvEst = []
    for i in range(1,longitud,5):
        datos = vector[i:i+4]
        media = sum(vector[i:i+4]) / 5
        suma_diferencias_cuadradas = sum([(dato - media) ** 2 for dato in datos])
        varianza = suma_diferencias_cuadradas / len(datos)
        desviacion_estandar = math.sqrt(varianza)
        desvEst.append(desviacion_estandar)
    return desvEst


#METODO PARA CALCULAR RMS (Raiz Media Cuadratica) EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE RMS (ROOT MEAN SQUARE) IN EACH SPECTRUM WITH A 5 "WINDOW"
def mediaCuadratica(vector):
    longitud = len(vector)
    medCuadratica = []

    for i in range(1,longitud,5):
        datos = vector[i:i+4]
        suma_diferencias_cuadradas = sum([x ** 2 for x in datos])
        media_cuadratica = suma_diferencias_cuadradas / len(datos)
        rms = math.sqrt(media_cuadratica)
        medCuadratica.append(rms)

    return medCuadratica

#METODO PARA CALCULAR LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE GEOMETRIC MEAN IN EACH SPECTRUM WITH A 5 "WINDOW"
def mediaGeo(vector):
    longitud = len(vector)
    mGeo = []

    for i in range(1,longitud,5):
        producto = np.prod(vector[i:i+4])
        res = math.pow(producto, 1/5)
        mGeo.append(res)
    return mGeo

#METODO PARA CALCULAR LAS VARIANZAS EN CADA ESPECTRO CON UNA VENTANA DE 5
#METHOD FOR CALCULATE THE VARIANCE IN EACH SPECTRUM WITH A 5 "WINDOW"
def varianza(vector):
    longitud = len(vector)
    var = []
    for i in range(1,longitud,5):
        varianza = np.var(vector[i:i+4])
        var.append(varianza)

    return var

#CARACTERISTICAS DE UNA IMAGEN
#IMAGE CHARACTERISTICS
def procesamientoImagen(imagen):
    #SEPARA LA IMAGEN LOS 3 ESPECTROS
    #SEPARE THE IMAGE IN 3 SPECTRAMES
    b, g, r = cv2.split(imagen)

    #SE CONVIERTE CADA ESPECTRO EN UN VECTOR
    #CONVERTS EACH SPECTRUM IN A VECTOR
    vectorB = b.ravel()
    vectorG = g.ravel()
    vectorR = r.ravel()

    # CALCULO DE LAS MEDIA EN CADA ESPECTRO
    # MEAN OF EACH SPECTRUM
    mediaB = media(vectorB)
    mediaG = media(vectorG)
    mediaR = media(vectorR)

    # CALCULO DE LAS MODAS EN CADA ESPECTRO
    # MODE OF EACH SPECTRUM
    modaB = moda(vectorB)
    modaG = moda(vectorG)
    modaR = moda(vectorR)

    # CALCULO DE LAS MEDIANAS EN CADA ESPECTRO
    # MEDIAN OF EACH SPECTRUM
    medianaB = mediana(vectorB)
    medianaG = mediana(vectorG)
    medianaR = mediana(vectorR)

    # CALCULO DE LAS DESVIACIONES ESTANDAR EN CADA ESPECTRO
    # STANDAR DEVIATION OF EACH SPECTRUM
    desvEstandarB = desvEstandar(vectorB)
    desvEstandarG = desvEstandar(vectorG)
    desvEstandarR = desvEstandar(vectorR)

    # CALCULO DE LAS VARIANZAS EN CADA ESPECTRO
    # VARIANCE OF EACH SPECTRUM
    varianzaB = varianza(vectorB)
    varianzaG = varianza(vectorG)
    varianzaR = varianza(vectorR)

    # CALCULO DE LAS MEDIAS GEOMETRICAS EN CADA ESPECTRO
    # GEOMETRIC MEAN OF EACH SPECTRUM
    mediaGeoB = mediaGeo(vectorB)
    mediaGeoG = mediaGeo(vectorG)
    mediaGeoR = mediaGeo(vectorR)

    # CALCULO DE LAS MEDIAS CUADRATICAS EN CADA ESPECTRO
    # RMS OF EACH SPECTRUM
    mediaCuaB = mediaCuadratica(vectorB)
    mediaCuaG = mediaCuadratica(vectorG)
    mediaCuaR = mediaCuadratica(vectorR)

    # UNION DE ESPECTROS POR FUNCION ESTADISTICA
    # UNION OF SPECTRAMES BY STATISTICAL FUNCTION
    mediaEspectrosUnidos = unirEspectrosFuncion(mediaB, mediaR, mediaG)
    modaEspectrosUnidos = unirEspectrosFuncion(modaB, modaR, modaG)
    medianaEspectrosUnidos = unirEspectrosFuncion(medianaB, medianaR, medianaG)
    desvEstandarEspectrosUnidos = unirEspectrosFuncion(desvEstandarB, desvEstandarR, desvEstandarG)
    varianzaEspectrosUnidos = unirEspectrosFuncion(varianzaB, varianzaR, varianzaG)
    mediaGeoEspectosUnidos = unirEspectrosFuncion(mediaGeoB, mediaGeoR, mediaGeoG)
    mediaCuaEspectrosUnidos = unirEspectrosFuncion(mediaCuaB, mediaCuaR, mediaCuaG)

    caracteristicas = mediaEspectrosUnidos + modaEspectrosUnidos + medianaEspectrosUnidos + desvEstandarEspectrosUnidos + varianzaEspectrosUnidos + mediaCuaEspectrosUnidos + mediaGeoEspectosUnidos

    return caracteristicas


#METODO PARA UNIR EN UN SOLO VECTOR LOS VECTORES DE LOS ESPECTROS
#METHOD TO JOIN THE SPECTRUM VECTORS IN A SINGLE VECTOR
def unirEspectrosFuncion(v1, v2, v3):
    caracteristicasImagen = v1 + v2 + v3
    return caracteristicasImagen

#METODO PARA AJUSTAR EL BRILLO
#METHOD TO ADJUST THE BRIGHTNESS
def adjust_brightness(img, brightness_factor):
    """Ajusta el brillo de una imagen"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * brightness_factor, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


with open('knn_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# CARGAR LA FUNCIÓN ENTRENADA
# LOAD THE TRAINED FUNCTION
CaracteristicasSuperior = []
CaracteristicasInferior = []

# INICIAR LA CÁMARA
# START THE CAMERA
cap = cv2.VideoCapture(0)

while(True):
    # CAPTURAR UNA IMAGEN DE LA CÁMARA
    # CAPTURE A FRAME
    frames = []
    for i in range(2):
        ret, frame = cap.read()
        img = cv2.resize(frame, (20, 20))
        frames.append(img)


    # PREPROCESAR LA IMAGEN
    # PREPROCESING THE FRAME
    for i in range(2):
        imagen = frames[i]

        image = adjust_brightness(imagen, 1.5)

        # DIVIDIR LA IMAGEN EN 2
        # DIVIDE THE FRAME IN 2
        height, width = imagen.shape[:2]


        # CALCULAR EL CENTRO DE LA IMAGEN
        # CALCULATE THE CENTER OF THE FRAME
        center = (width // 2, height // 2)

        # SELECCIONAR UN ÁREA DE 14X14 PIXELES EN EL CENTRO DE LA IMAGEN
        # SELECT A 14X14 PIXEL AREA IN THE CENTER OF THE FRAME
        area = img[center[1] - 7:center[1] + 7, center[0] - 7:center[0] + 7]
        area = cv2.resize(area, (20, 20))



        # VECTOR DE CARACTERISTICAS DE LA IMAGEN
        # VECTOR IMAGE CHARACTERISTICS
        caracteristicasT = procesamientoImagen(area)


        CaracteristicasSuperior.append(caracteristicasT)




    # UTILIZAR LA FUNCIÓN ENTRENADA PARA PREDECIR EL COLOR
    # USE THE TRAINED FUNCTION TO PREDICT THE COLOR

    prediction = clf.predict(CaracteristicasSuperior)

    # IMPRIMIR LAS ULTIMAS 3 ETIQUETAS
    # PRINT THE LAST 3 PREDICTIONS


    # MOSTRAR LA IMAGEN Y RESULTADO
    # SHOW THE IMAGE AND THE PREDICTION
    cv2.imshow("Camera", frame)
    print("                              ")
    print("-----------------------------")
    print("COLORES DETECTADOS")
    print("COLORES DETECTADOS:", prediction[-2:])

    print("                              ")
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    label.config(text="Colores detectados: {}".format(prediction[-2:]))
    root.update()

    #time.sleep(3)

# CERRAR LA CAMARA
# CLOSE THE CAMERA
root.mainloop()
cap.release()
cv2.destroyAllWindows()