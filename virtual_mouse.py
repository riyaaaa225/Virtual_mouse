import cv2
import mediapipe as mp
import pyautogui
import threading
import math
from deepface import DeepFace

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()

frame_count = 0
emotion = "neutral"

# Emotion Detection Thread
def detect_emotion(frame):
    global emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
    except:
        pass

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand tracking
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Get Index Finger Tip (8) and Thumb Tip (4) positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Convert to pixel coordinates
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Move cursor to index finger position
            pyautogui.moveTo(ix, iy)

            # Calculate distance between thumb & index finger
            distance = math.hypot(tx - ix, ty - iy)

            # If distance is very small (pinch), perform click
            if distance < 30:
                pyautogui.click()
                print("Click!")

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Run emotion detection every 10th frame
    frame_count += 1
    if frame_count % 10 == 0:
        threading.Thread(target=detect_emotion, args=(frame,)).start()
    
    cv2.imshow("Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

