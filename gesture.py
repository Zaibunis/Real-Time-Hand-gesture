import cv2
import mediapipe as mp
import streamlit as st



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Function to detect peace sign ✌️
def detect_peace_sign(landmarks):
    """Detects a 'peace sign' (✌️)."""
    index_finger = landmarks[8]   
    middle_finger = landmarks[12]
    ring_finger = landmarks[16]
    pinky_finger = landmarks[20]

    index_extended = index_finger[1] < landmarks[6][1]  # Index finger above knuckle
    middle_extended = middle_finger[1] < landmarks[10][1]
    ring_curled = ring_finger[1] > landmarks[14][1]
    pinky_curled = pinky_finger[1] > landmarks[18][1]

    return index_extended and middle_extended and ring_curled and pinky_curled

# Function to detect thumbs up 👍
def detect_thumbs_up(landmarks):
    """Detects a 'thumbs up' (👍)."""
    thumb_tip = landmarks[4]  # (x, y)
    thumb_base = landmarks[3]  # Base of the thumb

    thumb_extended = thumb_tip[1] < thumb_base[1]  # Thumb is above its base
    curled_fingers = [8, 12, 16, 20]  # Index, middle, ring, pinky
    fingers_curled = all(landmarks[finger][1] > landmarks[finger - 2][1] for finger in curled_fingers)

    return thumb_extended and fingers_curled

# Function to detect Saranghae (사랑해) ❤️
def detect_saranghae(landmarks):
    """Detects the 'Saranghae' (finger heart ❤️) gesture."""
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_finger = landmarks[12]
    ring_finger = landmarks[16]
    pinky_finger = landmarks[20]

    # Check if thumb & index finger tips are close (forming a heart)
    thumb_index_close = abs(thumb_tip[0] - index_tip[0]) < 0.05 and abs(thumb_tip[1] - index_tip[1]) < 0.05

    # Other fingers should be curled
    middle_curled = middle_finger[1] > landmarks[10][1]
    ring_curled = ring_finger[1] > landmarks[14][1]
    pinky_curled = pinky_finger[1] > landmarks[18][1]

    return thumb_index_close and middle_curled and ring_curled and pinky_curled

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Detect and display gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = {i: (lm.x, lm.y) for i, lm in enumerate(hand_landmarks.landmark)}

            if detect_peace_sign(landmarks):
                cv2.putText(frame, "Peace ✌️", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            elif detect_thumbs_up(landmarks):
                cv2.putText(frame, "Thumbs Up 👍", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            elif detect_saranghae(landmarks):
                cv2.putText(frame, "Saranghae ❤️", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    # Display Output
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Streamlit UI Config
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Sidebar Controls
st.sidebar.title("Hand Gesture Recognition 🖐️")
st.sidebar.write("Control the real-time hand gesture detection system.")

# Initialize session state for controlling the camera
if "running" not in st.session_state:
    st.session_state.running = False

def start_detection():
    st.session_state.running = True



# Buttons to start/stop detection
st.sidebar.button("Start Detection", on_click=start_detection)


# Mediapipe Hand Detection Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Streamlit Main App
st.title("🤖 Real-Time Hand Gesture Recognition")
st.write("This app detects hand gestures in real-time using OpenCV and MediaPipe.")

# Camera Input
frame_placeholder = st.empty()
detected_gesture_placeholder = st.empty()

# OpenCV Video Capture
def detect_hand_gestures():
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_gesture_placeholder.write("🖐️ Hand Detected!")
        else:
            detected_gesture_placeholder.write("No hand detected.")
        
        frame_placeholder.image(frame, channels="RGB")
    
    cap.release()

# Run detection when session state is True
if st.session_state.running:
    detect_hand_gestures()


