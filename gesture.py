# type: ignore

import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np
import io
import tempfile
import os
import time



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




# Streamlit setup with better UI
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #2c3e50;
        padding: 20px;
    }
    .gesture-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        text-align: center;
    }
    .detected {
        background-color: #2ecc71;
        color: white;
    }
    .not-detected {
        background-color: #95a5a6;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='title'>✨ Hand Gesture Recognition ✨</h1>", unsafe_allow_html=True)

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Webcam", "📤 Upload Image/Video"])

# Create columns for layout
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Gesture Status")
    peace_status = st.empty()
    thumbs_status = st.empty()
    heart_status = st.empty()
    
    st.markdown("### Instructions")
    st.markdown("""
    Try these gestures:
    - ✌️ Peace Sign
    - 👍 Thumbs Up
    - ❤️ Saranghae (Finger Heart)
    """)

def process_frame(frame):
    """Process a single frame and return detected gestures"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    detected_peace = False
    detected_thumbs = False
    detected_heart = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            
            detected_peace = detect_peace_sign(landmarks)
            detected_thumbs = detect_thumbs_up(landmarks)
            detected_heart = detect_saranghae(landmarks)
    
    return frame, detected_peace, detected_thumbs, detected_heart

def update_gesture_status(peace, thumbs, heart):
    """Update the gesture status displays"""
    peace_status.markdown(
        f"<div class='gesture-box {'detected' if peace else 'not-detected'}'>"
        f"✌️ Peace Sign: {'Detected!' if peace else 'Not Detected'}</div>", 
        unsafe_allow_html=True
    )
    thumbs_status.markdown(
        f"<div class='gesture-box {'detected' if thumbs else 'not-detected'}'>"
        f"👍 Thumbs Up: {'Detected!' if thumbs else 'Not Detected'}</div>", 
        unsafe_allow_html=True
    )
    heart_status.markdown(
        f"<div class='gesture-box {'detected' if heart else 'not-detected'}'>"
        f"❤️ Saranghae: {'Detected!' if heart else 'Not Detected'}</div>", 
        unsafe_allow_html=True
    )

with col1:
    with tab1:
        # Webcam Implementation
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("No webcam found. Try uploading an image or video instead.")
            else:
                frame_window = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, peace, thumbs, heart = process_frame(frame)
                    update_gesture_status(peace, thumbs, heart)
                    frame_window.image(frame, channels="BGR", use_container_width=True)

        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

    with tab2:
        # File upload implementation
        upload_type = st.radio("Choose upload type:", ["Image", "Video"])
        
        if upload_type == "Image":
            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                frame, peace, thumbs, heart = process_frame(frame)
                update_gesture_status(peace, thumbs, heart)
                st.image(frame, channels="BGR", use_container_width=True)
        
        else:  # Video
            uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                frame_window = st.empty()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame, peace, thumbs, heart = process_frame(frame)
                    update_gesture_status(peace, thumbs, heart)
                    frame_window.image(frame, channels="BGR", use_container_width=True)
                    time.sleep(0.1)  # Add small delay to make video playback smoother
                
                cap.release()
                os.unlink(tfile.name)

st.markdown("""
---
**Note:** 
- Webcam access might be limited on Streamlit Cloud
- For best results with webcam, run locally using `streamlit run gesture.py`
- Image and video upload features work on both local and cloud deployments
""")

