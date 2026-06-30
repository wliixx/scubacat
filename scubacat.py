import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)
mp_draw = vision.drawing_utils

video_path = 'cat.mp4'
cap_cat = None

cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0
if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    ret, frame = cap.read()          # читаем кадр
    if not ret:                     # если кадр не прочитался — выходим
        break

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
  
