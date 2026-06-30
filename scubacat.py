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
timestamp_ms = 0
video_play = False
# функция, проверяющая, находятся ли две точки рядом в пространстве для рук
def close_points(p1,p2, limit=0.1):
    distantion = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return distantion < limit

# функция, проверяющая, сжат ли кулак
def is_fist_clenched(hui):
    konchiki = [8,12,16,20]
    bases = hui[0]
    close = [close_points(hui[k], bases, 0.25) for k in konchiki]
    return all(close)

# функция, проверяющая, открыта ли ладонь
def openhand(hui):
    bases_konchiki =[(8,6),(12,10),(16,14),(20,18)]
    return all(hui[k].y < hui[b].y for k, b in bases_konchiki)

# открытие видео и отслеживание жестов
while True:
    ret, frame = cap.read()          
    if not ret:                     
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms += 33
    result = detector.detect_for_video(mp_image,timestamp_ms)
    
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
  