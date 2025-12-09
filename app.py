import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import numpy as np
import av
import threading
import time
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==========================================
# Configuration
# ==========================================
FACE_MODEL_PATH = 'weights/yolov8n-face-lindevs.pt'
HAND_MODEL_PATH = 'hand_yolov8n.pt'
KNOWN_DB_DIR = "people faces"
CONTACT_THRESHOLD = 250
COOLDOWN_SECONDS = 3

# ==========================================
# Global/Cached Models
# ==========================================
# We load models globally or cached to avoid reloading on every frame/session
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def load_models():
    face = YOLO(FACE_MODEL_PATH) if os.path.exists(FACE_MODEL_PATH) else YOLO('yolov8n.pt')
    hand = YOLO(HAND_MODEL_PATH)
    return face, hand

try:
    FACE_MODEL, HAND_MODEL = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================================
# Identification Logic
# ==========================================
def identify_face(face_img, callback_success, callback_fail):
    """
    Runs in a separate thread. 
    callback_success(name, match_img_path)
    callback_fail()
    """
    try:
        # Save temp for deepface (optional, but robust)
        # DeepFace.find can accept numpy, but we'll pass numpy directly
        results = DeepFace.find(
            img_path=face_img, 
            db_path=KNOWN_DB_DIR, 
            model_name="Facenet512", # Fast & Good
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False, 
            silent=True
        )
        
        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            match_path = best_match['identity']
            
            # Extract Name
            path_parts = os.path.normpath(match_path).split(os.sep)
            db_folder_name = os.path.basename(os.path.normpath(KNOWN_DB_DIR))
            parent_folder = path_parts[-2]
            
            if parent_folder == db_folder_name:
                name = os.path.splitext(path_parts[-1])[0]
            else:
                name = parent_folder
                
            callback_success(name, match_path)
        else:
            callback_fail()
            
    except Exception as e:
        print(f"ID Error: {e}")
        callback_fail()

# ==========================================
# Video Processor
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_contact_time = 0
        self.identified_name = None
        self.matched_image = None # This will be a numpy array (BGR)
        self.identifying = False
        
        # Load match image helper
        self.lock = threading.Lock()

    def set_identity(self, name, match_path):
        with self.lock:
            self.identified_name = name
            self.identifying = False
            
            # Load the matched image for overlay
            if match_path and os.path.exists(match_path):
                img = cv2.imread(match_path)
                if img is not None:
                    # Resize for overlay (e.g., 100x100)
                    self.matched_image = cv2.resize(img, (120, 120))
    
    def set_unknown(self):
        with self.lock:
            self.identified_name = "Unknown"
            self.matched_image = None
            self.identifying = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Inference
        results_face = FACE_MODEL(img, verbose=False, conf=0.5)
        results_hand = HAND_MODEL(img, verbose=False, conf=0.5)
        
        # 2. Draw Faces
        face_boxes = []
        for box in results_face[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            face_boxes.append((x1, y1, x2, y2))
            
        # 3. Draw Hands & Contact Logic
        hand_centers = []
        for box in results_hand[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            hand_centers.append(center)
        
        contact = False
        if len(hand_centers) >= 2:
            import math
            min_dist = float('inf')
            closest_pair = None
            
            for i in range(len(hand_centers)):
                for j in range(i + 1, len(hand_centers)):
                    d = math.sqrt((hand_centers[i][0] - hand_centers[j][0])**2 + (hand_centers[i][1] - hand_centers[j][1])**2)
                    if d < min_dist: 
                        min_dist = d
                        closest_pair = (hand_centers[i], hand_centers[j])
            
            if min_dist < CONTACT_THRESHOLD:
                contact = True
                cv2.line(img, closest_pair[0], closest_pair[1], (0, 0, 255), 4)
                cv2.putText(img, "CONTACT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 4. Identification Trigger
        if contact:
            now = time.time()
            if not self.identifying and (now - self.last_contact_time > COOLDOWN_SECONDS):
                # Find largest face
                best_face = None
                max_area = 0
                h, w, _ = img.shape
                
                for (fx1, fy1, fx2, fy2) in face_boxes:
                    area = (fx2 - fx1) * (fy2 - fy1)
                    if area > max_area:
                        max_area = area
                        # Ensure bounds
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(w, fx2), min(h, fy2)
                        best_face = img[fy1:fy2, fx1:fx2].copy()
                
                if best_face is not None and best_face.size > 0:
                    self.identifying = True
                    self.last_contact_time = now
                    
                    # Start thread
                    t = threading.Thread(target=identify_face, args=(best_face, self.set_identity, self.set_unknown))
                    t.start()
        
        # 5. Overlay Results (Thread-safe reading)
        with self.lock:
            if self.identifying:
                cv2.putText(img, "Identifying...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if self.identified_name:
                text = f"ID: {self.identified_name}"
                color = (0, 255, 0) if self.identified_name != "Unknown" else (0, 0, 255)
                cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Overlay Matched Image (PIP)
                if self.matched_image is not None:
                    h_ov, w_ov, _ = self.matched_image.shape
                    h_img, w_img, _ = img.shape
                    
                    # Bottom-Right corner
                    y_offset = h_img - h_ov - 20
                    x_offset = w_img - w_ov - 20
                    
                    if y_offset > 0 and x_offset > 0:
                        img[y_offset:y_offset+h_ov, x_offset:x_offset+w_ov] = self.matched_image
                        cv2.rectangle(img, (x_offset, y_offset), (x_offset+w_ov, y_offset+h_ov), (255,255,255), 2)
                        cv2.putText(img, "MATCH", (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Main UI
# ==========================================
st.set_page_config(page_title="Amrit Sparsh - Recognition", layout="wide")
st.title("🤝 Amrit Sparsh: Handshake Recognition")
st.write("Ensuring safe and recognized contacts.")

webrtc_streamer(
    key="handshake-cam", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
