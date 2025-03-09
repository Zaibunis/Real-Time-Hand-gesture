# type: ignore

import cv2
import mediapipe as mp
import streamlit as st



# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils



# Function to detect peace sign ✌
def detect_peace_sign(landmarks):
    """Detects a 'peace sign' (✌)."""
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

# Function to detect Saranghae (사랑해) ❤
def detect_saranghae(landmarks):
    """Detects the 'Saranghae' (finger heart ❤) gesture."""
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


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit setup
st.title("Hand Gesture Recognition")

# Initialize the webcam capture
cap = cv2.VideoCapture(1)  # This line initializes the webcam feed

# Check if webcam is accessible
if not cap.isOpened():
    st.error("Failed to access the webcam. Please check the camera connection or close other applications using the camera.")
else:
    # Streamlit empty container for the webcam feed
    frame_window = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Detect and display landmarks (hand gestures)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame directly in the Streamlit app
        frame_window.image(frame, caption="Webcam Feed", channels="BGR", use_container_width=True)

    # Release the capture once finished
    cap.release()

