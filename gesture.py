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
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Failed to access the webcam. Please check the camera connection.")
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_window = st.empty()
    
    with col2:
        st.markdown("### Detected Gestures")
        gesture_status = st.empty()
        
        st.markdown("""
        ### Try these gestures:
        - ✌️ Peace Sign: Index and middle fingers up
        - 👍 Thumbs Up: Only thumb up
        - ❤️ Saranghae: Thumb and index finger heart
        """)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Initialize gesture states
        detected_gestures = []
        height, width = frame.shape[:2]

        # Process hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks with better visibility
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Convert landmarks to the format our detection functions expect
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                
                # Actively check for each gesture
                try:
                    # Peace Sign Detection
                    if detect_peace_sign(landmarks):
                        detected_gestures.append("✌️ Peace Sign")
                        cv2.putText(frame, "Peace Sign!", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Draw visual feedback
                        index_pos = (int(landmarks[8][0] * width), int(landmarks[8][1] * height))
                        middle_pos = (int(landmarks[12][0] * width), int(landmarks[12][1] * height))
                        cv2.circle(frame, index_pos, 8, (0, 255, 0), -1)
                        cv2.circle(frame, middle_pos, 8, (0, 255, 0), -1)
                    
                    # Thumbs Up Detection
                    if detect_thumbs_up(landmarks):
                        detected_gestures.append("👍 Thumbs Up")
                        cv2.putText(frame, "Thumbs Up!", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Draw visual feedback
                        thumb_pos = (int(landmarks[4][0] * width), int(landmarks[4][1] * height))
                        cv2.circle(frame, thumb_pos, 8, (0, 255, 0), -1)
                    
                    # Saranghae Detection
                    if detect_saranghae(landmarks):
                        detected_gestures.append("❤️ Saranghae")
                        cv2.putText(frame, "Saranghae!", (10, 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # Draw visual feedback
                        thumb_pos = (int(landmarks[4][0] * width), int(landmarks[4][1] * height))
                        index_pos = (int(landmarks[8][0] * width), int(landmarks[8][1] * height))
                        cv2.line(frame, thumb_pos, index_pos, (0, 0, 255), 2)

                except Exception as e:
                    st.error(f"Detection error: {str(e)}")

        # Update status with active feedback
        if detected_gestures:
            gesture_text = "\n".join([
                f"<div style='padding:10px; background-color:#2ecc71; color:white; "
                f"border-radius:5px; margin:5px; font-size:18px; font-weight:bold;'>"
                f"Detected: {gesture}</div>"
                for gesture in detected_gestures
            ])
        else:
            if results.multi_hand_landmarks:
                gesture_text = "<div style='padding:10px; background-color:#f39c12; color:white; " \
                             "border-radius:5px; margin:5px;'>Hand detected! Try making a gesture</div>"
            else:
                gesture_text = "<div style='padding:10px; background-color:#95a5a6; color:white; " \
                             "border-radius:5px; margin:5px;'>Show your hand to the camera</div>"
        
        gesture_status.markdown(gesture_text, unsafe_allow_html=True)

        # Add debug info
        cv2.putText(frame, f"Detected gestures: {len(detected_gestures)}", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        frame_window.image(frame, channels="BGR", use_container_width=True)

    # Release the capture once finished
    cap.release()

# Add stop button
if st.button('Stop'):
    cap.release()
    st.stop()

