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




# Streamlit setup with better UI
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Add custom CSS
# ... (keep existing CSS) ...

st.markdown("<h1 class='title'>✨ Hand Gesture Recognition ✨</h1>", unsafe_allow_html=True)

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

with col1:
    # Try multiple camera indices
    camera_found = False
    for i in range(-1, 3):  # Try indices -1, 0, 1, 2
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_found = True
                st.success(f"Successfully connected to camera {i}")
                break
        except Exception as e:
            continue

    if not camera_found:
        st.error("No camera found. Please check your camera connection and permissions.")
        st.info("If you're running this on Streamlit Cloud, please note that camera access might be restricted.")
    else:
        frame_window = st.empty()
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break

                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    # Detect and display landmarks
                    detected_peace = False
                    detected_thumbs = False
                    detected_heart = False

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            # Convert landmarks to list format
                            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                            
                            # Check for gestures
                            detected_peace = detect_peace_sign(landmarks)
                            detected_thumbs = detect_thumbs_up(landmarks)
                            detected_heart = detect_saranghae(landmarks)

                    # Update gesture status boxes
                    peace_status.markdown(
                        f"<div class='gesture-box {'detected' if detected_peace else 'not-detected'}'>"
                        f"✌️ Peace Sign: {'Detected!' if detected_peace else 'Not Detected'}</div>", 
                        unsafe_allow_html=True
                    )
                    thumbs_status.markdown(
                        f"<div class='gesture-box {'detected' if detected_thumbs else 'not-detected'}'>"
                        f"👍 Thumbs Up: {'Detected!' if detected_thumbs else 'Not Detected'}</div>", 
                        unsafe_allow_html=True
                    )
                    heart_status.markdown(
                        f"<div class='gesture-box {'detected' if detected_heart else 'not-detected'}'>"
                        f"❤️ Saranghae: {'Detected!' if detected_heart else 'Not Detected'}</div>", 
                        unsafe_allow_html=True
                    )

                    # Display the frame
                    frame_window.image(frame, channels="BGR", use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break

        except Exception as e:
            st.error(f"Error in main loop: {str(e)}")
        
        finally:
            cap.release()

# Add note about Streamlit Cloud
st.markdown("""
---
**Note:** This application requires camera access. If you're running this on Streamlit Cloud, 
you might experience limited functionality due to platform restrictions. For best results, 
run this application locally using `streamlit run gesture.py`
""")

