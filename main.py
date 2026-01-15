import cv2
import numpy as np
from google import genai
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
hand1 = 0
hand2 = 0
nums = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789', '!@#$%^&*()[];:"<>?/,.']
let_click1 = True
let_click2 = False
let_click3 = False
let_click4 = False
let_click5 = False
c = 0
text = ''
result = ''
res = ''
c1 = 0
r = []
with open('apikey.txt','r') as file:
    content = file.read()
try:
    client = genai.Client(api_key=content)
except Exception as e:
    print(f"Error initializing client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set.")
    exit()
model_name = 'gemini-2.5-flash-lite'

def print_result(result, output_image, timestamp_ms):
    global x_coord1, y_coord1, z_coord1, hand1, hand2, x_coord2, y_coord2, z_coord2
    # set coords when a hand is found, otherwise reset and mark absent
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        if len(result.handedness) == 2:
            hand1 = 1
            hand2 = 1
        elif len(result.handedness) == 1:
            if result.handedness[0][0].category_name == 'Left':
                hand1 = 0
                hand2 = 1
            else:
                hand1 = 1
                hand2 = 0
        else:
            hand1 = 0
            hand2 = 0
        if len(result.handedness) == 1:
            if hand1 == 1:
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            elif hand2 == 1:
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
                x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif len(result.handedness) == 2:
            if result.handedness[0][0].category_name == 'Right':
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                lm_list = result.hand_landmarks[1]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
            else:
                lm_list = result.hand_landmarks[1]
                for i, lm in enumerate(lm_list):
                    x_coord1[i] = lm.x
                    y_coord1[i] = lm.y
                    z_coord1[i] = lm.z
                lm_list = result.hand_landmarks[0]
                for i, lm in enumerate(lm_list):
                    x_coord2[i] = lm.x
                    y_coord2[i] = lm.y
                    z_coord2[i] = lm.z
        present = True
    else:
        x_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        z_coord1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        x_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        y_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        z_coord2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        present = False
    videoFeed(output_image, present)

def videoFeed(img, present):
    global c, x_coord1, y_coord1, z_coord1, nums, let_click1, x_coord2, y_coord2, z_coord2, hand1, hand2, let_click2, text, let_click3, result, res, c1, model_name, let_click4, let_click5
    n_frame = img.numpy_view()
    new_frame = np.copy(n_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
    distance1 = ((x_coord1[4]-x_coord1[8])**2 + (y_coord1[4]-y_coord1[8])**2 + (z_coord1[4]-z_coord1[8])**2)**0.5
    distance2 = ((x_coord2[4]-x_coord2[8])**2 + (y_coord2[4]-y_coord2[8])**2 + (z_coord2[4]-z_coord2[8])**2)**0.5
    distance3 = ((x_coord1[4]-x_coord1[12])**2 + (y_coord1[4]-y_coord1[12])**2 + (z_coord1[4]-z_coord1[12])**2)**0.5
    distance4 = ((x_coord2[4]-x_coord2[12])**2 + (y_coord2[4]-y_coord2[12])**2 + (z_coord2[4]-z_coord2[12])**2)**0.5
    distance5 = ((x_coord1[4]-x_coord1[16])**2 + (y_coord1[4]-y_coord1[16])**2 + (z_coord1[4]-z_coord1[16])**2)**0.5
    if present and hand1 == 1:
        cv2.circle(new_frame, (int(x_coord1[4]*new_frame.shape[1]), int(y_coord1[4]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.circle(new_frame, (int(x_coord1[8]*new_frame.shape[1]), int(y_coord1[8]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord1[4]*new_frame.shape[1]), int(y_coord1[4]*new_frame.shape[0])), (int(x_coord1[8]*new_frame.shape[1]), int(y_coord1[8]*new_frame.shape[0])), (255, 255, 255), 2)
        cv2.circle(new_frame, (int(x_coord1[12]*new_frame.shape[1]), int(y_coord1[12]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord1[4]*new_frame.shape[1]), int(y_coord1[4]*new_frame.shape[0])), (int(x_coord1[12]*new_frame.shape[1]), int(y_coord1[12]*new_frame.shape[0])), (255, 255, 255), 2)
        cv2.circle(new_frame, (int(x_coord1[16]*new_frame.shape[1]), int(y_coord1[16]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord1[4]*new_frame.shape[1]), int(y_coord1[4]*new_frame.shape[0])), (int(x_coord1[16]*new_frame.shape[1]), int(y_coord1[16]*new_frame.shape[0])), (255, 255, 255), 2)
    if present and hand2 == 1:
        cv2.circle(new_frame, (int(x_coord2[4]*new_frame.shape[1]), int(y_coord2[4]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.circle(new_frame, (int(x_coord2[8]*new_frame.shape[1]), int(y_coord2[8]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord2[4]*new_frame.shape[1]), int(y_coord2[4]*new_frame.shape[0])), (int(x_coord2[8]*new_frame.shape[1]), int(y_coord2[8]*new_frame.shape[0])), (255, 255, 255), 2)
        cv2.circle(new_frame, (int(x_coord2[12]*new_frame.shape[1]), int(y_coord2[12]*new_frame.shape[0])), 5, (255, 255, 255), -1)
        cv2.line(new_frame, (int(x_coord2[4]*new_frame.shape[1]), int(y_coord2[4]*new_frame.shape[0])), (int(x_coord2[12]*new_frame.shape[1]), int(y_coord2[12]*new_frame.shape[0])), (255, 255, 255), 2)
    flip = cv2.flip(new_frame, 1)
    cv2.putText(flip, f'Distance(Change) : {distance2:.3f}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(flip, f'Distance(Click) : {distance1:.3f}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(flip, f'Distance(Input) : {distance3:.3f}', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    if present and hand2:
        angle = np.arctan((y_coord2[4]-y_coord2[8])/(x_coord2[4]-x_coord2[8]))*180/3.14
        if angle < 0:
            angle = 180 + angle
    else:
        angle = 0
    cv2.putText(flip, f'Angle : {angle:.3f}', (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    if hand2 == 1:
        num = nums[c][int((angle/180)*len(nums[c]))]
    else:
        num = None
    cv2.putText(flip, f'Character : {num}', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(flip, (20, 110), (200, 130), (255, 255, 255), 1)
    if distance2 > 0.1:
        let_click1 = True
    if distance2 <= 0.05 and let_click1 and x_coord2[4] > 0.1 and x_coord2[4] < 0.9 and y_coord2[4] > 0.1 and y_coord2[4] < 0.9 and y_coord2[8] > 0.1 and y_coord2[8] < 0.9 and y_coord2[8] > 0.1 and y_coord2[8] < 0.9:
        if c < 2:
            c += 1
        else:
            c = 0
        let_click1 = False
    if distance1 > 0.1:
        let_click2 = True
    if distance1 <= 0.05 and let_click2 and x_coord1[4] > 0.1 and x_coord1[4] < 0.9 and y_coord1[4] > 0.1 and y_coord1[4] < 0.9 and y_coord1[8] > 0.1 and y_coord1[8] < 0.9 and y_coord1[8] > 0.1 and y_coord1[8] < 0.9:
        text = text + num
        let_click2 = False
    if distance3 > 0.1:
        let_click4 = True
    if distance3 <= 0.05 and let_click4 and x_coord1[4] > 0.1 and x_coord1[4] < 0.9 and y_coord1[4] > 0.1 and y_coord1[4] < 0.9 and y_coord1[12] > 0.1 and y_coord1[12] < 0.9 and y_coord1[12] > 0.1 and y_coord1[12] < 0.9:
        text = text[:len(text)-1]
        let_click4 = False
    if distance5 > 0.1:
        let_click5 = True
    if distance5 <= 0.05 and let_click5 and x_coord1[4] > 0.1 and x_coord1[4] < 0.9 and y_coord1[4] > 0.1 and y_coord1[4] < 0.9 and y_coord1[16] > 0.1 and y_coord1[16] < 0.9 and y_coord1[16] > 0.1 and y_coord1[16] < 0.9:
        text = text + ' '
        let_click5 = False
    if distance4 > 0.1:
        let_click3 = True
    if distance4 <= 0.05 and let_click3 and x_coord2[4] > 0.1 and x_coord2[4] < 0.9 and y_coord2[4] > 0.1 and y_coord2[4] < 0.9 and y_coord2[12] > 0.1 and y_coord2[12] < 0.9 and y_coord2[12] > 0.1 and y_coord2[12] < 0.9:
        try:
            response = client.models.generate_content(model=model_name, contents=(text+'keep your answer within 800 words'))
            text = ''
            result = response.text
        except Exception as e:
            print(f"An error occurred during content generation: {e}")
        let_click3 = False
    cv2.putText(flip, text, (23, 125), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    if result != '':
        if len(result) <= 400:
            size = 0.5
            l = 50
        elif len(result) > 400 and len(result) <= 800:
            size = 0.4
            l = 60
        elif len(result) >= 800:
            size = 0.3
            l = 70
        for i in range(0, len(result)-l, l):
            cv2.putText(flip, result[i:i+l], (20, 145+20*i//l), cv2.FONT_HERSHEY_COMPLEX, size, (255,255,255), 1)
    cv2.imshow('Live Video Feed', flip)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        
cap = cv2.VideoCapture(0)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5, num_hands=2,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
    
    last_timestamp_ms = 0
    

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= last_timestamp_ms:
            now_ms = last_timestamp_ms + 1
        last_timestamp_ms = now_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=gray)
        landmarker.detect_async(mp_image, last_timestamp_ms)

