import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO)
detector = vision.HandLandmarker.create_from_options(options)
            #mp_hands = mp.solution.hands
            #hands = mp_hands.Hands(
                #static_image_mode = False, # Режим для видео (False)
                #max_num_hands = 2,  # Максимальное количество рук для обнаружения
                #min_detection_confidence = 0.5, # Минимальная уверенность детекции.(уровень уверенности, что это рука, а не нога или еще чота)
                #min_tracking_confidence = 0.5 # Минимальная уверенность трекинга
            #) #- версия не подошла, ну ладно, пусть будет.

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
  