import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math 

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2, running_mode=vision.RunningMode.VIDEO, min_hand_detection_confidence = 0.1, min_tracking_confidence = 0.1, min_hand_presence_confidence = 0.1)
detector = vision.HandLandmarker.create_from_options(options)

video_path = 'cat.mp4'
cap_cat = None


cap = cv2.VideoCapture(0)
timestamp_ms = 0

# hand point connection. this is designed to draw the skeleton manually
HAND_CONNECTIONS = [     # связь точек руки, предназначено для того, чтобы нарисовать скелет вруную
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)]



# a function that checks whether two points are close together in the hand space
def close_points(p1,p2, limit=0.1):      # функция, проверяющая, находятся ли две точки рядом в пространстве для рук
    distantion = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    return distantion < limit

# a function that checks if the fist is clenched
def is_fist_clenched(lm):   # функция, проверяющая, сжат ли кулак        
    fingertips = [8,12,16,20]
    wrist = lm[0]
    close = [close_points(lm[k], wrist, 0.25) for k in fingertips]
    return all(close)

# a function that checks if the palm is open
def openhand(lm):   # функция, проверяющая, открыта ли ладонь
    wrist_fingertips =[(8,6),(12,10),(16,14),(20,18)]
    return all(lm[k].y < lm[b].y for k, b in wrist_fingertips)

# a function that draws a skeleton for a hand
def draw_landmarks(frame, landmarks):    # a function that draws a skeleton for a hand
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * w)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 225, 0), 2)
    for x,y in points:
        cv2.circle(frame,(x,y), 4, (0, 0, 255), -1)



# video opening and gesture tracking
while True:         # открытие видео и отслеживание жестов
    ret, frame = cap.read()          
    if not ret:                     
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms += 33
    result = detector.detect_for_video(mp_image,timestamp_ms)

    gesture_match = False
    
    if result.hand_landmarks:
        fist_found = False
        open_found = False
        for hand in result.hand_landmarks:
            draw_landmarks(frame, hand)
            if is_fist_clenched(hand):
                fist_found = True
            if openhand(hand):
                open_found = True
            if len(result.hand_landmarks) == 2 and fist_found and open_found:
                gesture_match = True

    if gesture_match:
        if cap_cat is None:
            cap_cat = cv2.VideoCapture(video_path)
        ret_cat, frame_cat = cap_cat.read()
        if not ret_cat:
            cap_cat.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_cat, frame_cat = cap_cat.read()
        if ret_cat:
            cv2.imshow('Cat', frame_cat)
    else:
       if cap_cat is not None:
            cap_cat.release()
            cap_cat = None
            cv2.destroyWindow('Cat')

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break