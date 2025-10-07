#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : openpose_for_image_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-29 21:50:17
"""
import time
import cv2 as cv
import numpy as np  # pour des opérations sur des tableaux et matrices
######################### Detection ##########################
# load the COCO class names charge les noms des classes d'objets a detecter
with open('object_detection_coco.txt', 'r') as f: class_names = f.read().split('\n')
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
# load the DNN modelimage
model = cv.dnn.readNet(model='frozen_inference_graph.pb', config='ssd_mobilenet_v2_coco.txt', framework='TensorFlow')

######################### openpose ##########################
#Définit un dictionnaire qui mappe les noms des parties du corps
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
          "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
          "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
          "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
#Définit les paires de connexions entre les parties du corps
POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
          ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
          ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
          ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
          ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
#Charge un modèle de détection de pose
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

def Target_Detection(image):
    image_height, image_width, _ = image.shape
    # create blob from image
    blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    model.setInput(blob)  #Envoie le blob en entrée au modèle
    output = model.forward()  #Exécute une passe avant sur le modèle pour obtenir les résultats de la détection.
    # loop over each of the detections
    for detection in output[0, 0, :, :]:
        # Extrait le niveau de confiance pour la détection actuelle.
        confidence = detection[2]
        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        if confidence > .4:
            # get the class id
            class_id = detection[1]
            # map the class id to the class
            class_name = class_names[int(class_id) - 1]
            if class_name == "stop sign":  # 
                print("Panneau 'Stop' détecté - arrêt de l'application")
                return None  # Retourne None pour indiquer l'arrêt de l'application

            color = COLORS[int(class_id)]
            # get the bounding box coordinates
            box_x = detection[3] * image_width # Coordonnée X en pixels du coin supérieur gauche
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            # draw a rectangle around each detected object
            cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            # put the class name text on the detected object
            cv.putText(image, class_name, (int(box_x), int(box_y - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def openpose(frame):
    frameHeight, frameWidth = frame.shape[:2]
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)) #Convertit l'image en un blob
    out = net.forward() #contient les heatmaps pour chaque point clé de la pose du corps
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert (len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way. (trouver la position la plus probable pour la partie du corps)
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3] #Redimensionne les coordonnées du point clé
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > 0.2 else None)
    for pair in POSE_PAIRS: #Boucle sur chaque paire de parties du corps qui doivent être connectées.
        partFrom = pair[0] #première partie de la paire
        partTo = pair[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]: #Vérifie les points pour les deux parties
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3) #Dessine une ligne verte entre les deux points dans l'image frame.
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED) # Dessine un petit cercle rouge rempli autour de la première partie de la paire
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    return frame


if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    cv_edition = cv.__version__
    if cv_edition[0] == '3': capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'XVID'))
    else: capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640) #Définit la largeur de la vidéo capturée à 640 pixels.
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    state=True
    while capture.isOpened():
        start = time.time() #enregistre l'heure de début de la boucle
        ret, frame = capture.read()
        action = cv.waitKey(10) & 0xFF
        if state==True: 
            frame = Target_Detection(frame) #Effectue la détection d'objets sur frame
            if frame is None:
                break  # Arrêtez la boucle si un panneau 'stop' est détecté
        else: frame = openpose(frame)
        end = time.time()
        fps = 1 / (end - start)
        text = "FPS : " + str(int(fps))
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)
        cv.imshow('frame', frame)
        if action == ord('q') or action == 113: break
        if action == ord('f'):state = not state
    capture.release()
    cv.destroyAllWindows()






