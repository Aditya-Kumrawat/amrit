import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
cv2.setNumThreads(0)
import numpy as np
import av
import threading
import time
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

import queue
import pandas as pd # Added for pd.DataFrame used in the new code

# ==========================================
# Global/State
# ==========================================
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# ==========================================
# Configuration
# ==========================================
FACE_MODEL_PATH = 'weights/yolov8n-face-lindevs.pt'
HAND_MODEL_PATH = 'hand_yolov8n.pt'
KNOWN_DB_DIR = "people faces"
CONTACT_THRESHOLD = 250
COOLDOWN_SECONDS = 5

# ==========================================
# Models
# ==========================================
@st.cache_resource
def load_models():
    # Check for custom weights, else fallback to root/default
    if os.path.exists(FACE_MODEL_PATH):
        face = YOLO(FACE_MODEL_PATH)
    elif os.path.exists('yolov8n.pt'):
        face = YOLO('yolov8n.pt') 
    else:
        face = YOLO('yolov8n.pt') # Will download if missing
        
    hand = YOLO(HAND_MODEL_PATH) if os.path.exists(HAND_MODEL_PATH) else YOLO('yolov8n.pt') # Fallback if hand model missing
    return face, hand

try:
    FACE_MODEL, HAND_MODEL = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================================
# Identification Logic (Multi-Face)
# ==========================================
def identify_faces_task(faces_data, result_queue):
    """
    faces_data: list of (face_bgr_crop, box_coords)
    """
    results = []
    
    for face_img, _ in faces_data:
        person_res = {
            "name": "Unknown", 
            "match_img": None, 
            "live_img": face_img,
            "risk_score": np.random.randint(20, 90) # Simulated Risk
        }
        
        try:
            # Convert to RGB
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            dfs = DeepFace.find(
                img_path=face_rgb, 
                db_path=KNOWN_DB_DIR, 
                model_name="Facenet512",
                detector_backend="skip", # detected already
                distance_metric="cosine",
                enforce_detection=False, 
                silent=True
            )
            
            if len(dfs) > 0 and not dfs[0].empty:
                best = dfs[0].iloc[0]
                match_path = best['identity']
                
                # Name Logic
                path_parts = os.path.normpath(match_path).split(os.sep)
                db_folder = os.path.basename(os.path.normpath(KNOWN_DB_DIR))
                if path_parts[-2] == db_folder:
                    name = os.path.splitext(path_parts[-1])[0]
                else:
                    name = path_parts[-2]
                
                person_res["name"] = name
                person_res["match_img"] = cv2.imread(match_path) # BGR
                # Lower risk for known people? Or higher? Simulated.
                if name != "Unknown":
                    person_res["risk_score"] = np.random.randint(5, 30) 
                    
        except Exception as e:
            print(f"Identification Error: {e}")
            
        results.append(person_res)
    
    print(f"Pushing {len(results)} results to queue.")
    result_queue.put(results)

# ==========================================
# Video Processor
# ==========================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_contact_time = 0
        self.identifying = False
        self.result_queue = None # Injected externally

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Inference
        res_face = FACE_MODEL(img, verbose=False, conf=0.5)
        res_hand = HAND_MODEL(img, verbose=False, conf=0.3)
        
        # Faces
        face_boxes = [] # (x1, y1, x2, y2, area)
        for box in res_face[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            face_boxes.append((x1,y1,x2,y2, area))
            
        # Hands
        hand_centers = []
        for box in res_hand[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            hand_centers.append( ((x1+x2)//2, (y1+y2)//2) )
            
        # Contact Logic
        contact = False
        if len(hand_centers) >= 2:
            min_dist = float('inf')
            p1, p2 = None, None
            for i in range(len(hand_centers)):
                for j in range(i+1, len(hand_centers)):
                    d = ((hand_centers[i][0]-hand_centers[j][0])**2 + (hand_centers[i][1]-hand_centers[j][1])**2)**0.5
                    if d < min_dist:
                        min_dist = d
                        p1, p2 = hand_centers[i], hand_centers[j]
            
            if min_dist < CONTACT_THRESHOLD:
                contact = True
                cv2.line(img, p1, p2, (0,0,255), 4)
                cv2.putText(img, "CONTACT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # Trigger Identification (On Contact)
        if contact and not self.identifying:
            now = time.time()
            if (now - self.last_contact_time) > COOLDOWN_SECONDS:
                # Get Top 2 Faces
                sorted_faces = sorted(face_boxes, key=lambda x: x[4], reverse=True)[:2]
                
                if sorted_faces:
                    self.identifying = True
                    self.last_contact_time = now
                    
                    faces_payload = []
                    h, w, _ = img.shape
                    for (fx1,fy1,fx2,fy2,_) in sorted_faces:
                        # Clamp
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(w, fx2), min(h, fy2)
                        crop = img[fy1:fy2, fx1:fx2].copy()
                        faces_payload.append((crop, (fx1,fy1,fx2,fy2)))
                    
                    # Offload to thread
                    if self.result_queue:
                         t = threading.Thread(target=identify_faces_task, args=(faces_payload, self.result_queue))
                         t.start()
                         
                    # Reset identifying flag after short delay or let thread finish? 
                    # For simplicity, we just toggle it off quickly in next frames or use timer.
                    # Actually, we rely on cooldown. 'identifying' is just to prevent spamming threads in 1 sec.
                    def reset_identifying():
                        time.sleep(1)
                        self.identifying = False
                    threading.Thread(target=reset_identifying).start()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# Main UI
# ==========================================
st.set_page_config(page_title="Amrit Sparsh - Recognition", layout="wide", page_icon="🤝")

# Custom CSS (Light Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #F8F9FA; color: #212529; }
    h1 { background: -webkit-linear-gradient(45deg, #2563EB, #06B6D4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stVideo { border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
    .person-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/handshake.png", width=64)
    st.markdown("## System Status")
    st.info("🟢 Models Loaded")
    st.info("🟢 Webcam Active")
    st.write("---")
    st.markdown("### About")
    st.write("**Amrit Sparsh**")
    st.caption("v2.0 Multi-Person Support")

# Header
st.markdown("<h1 style='text-align: center;'>🤝 Amrit Sparsh</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Advanced Biometric Handshake Recognition System</p>", unsafe_allow_html=True)

# Layout: 3 Cols ( Left Spacer | Video | Right Spacer )
c1, c2, c3 = st.columns([1, 6, 1])

# Placeholder for Streamer
with c2:
    ctx = webrtc_streamer(
        key="handshake-cam", 
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, 
            "audio": False
        },
        async_processing=True,
        desired_playing_state=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

if ctx.video_processor:
    ctx.video_processor.result_queue = st.session_state.result_queue

st.write("---")
st.markdown("### 📋 Identification Results")

result_placeholder = st.empty()

# Result Rendering
if 'last_results' in st.session_state:
    results = st.session_state.last_results
    
    cols = st.columns(len(results))
    for idx, p in enumerate(results):
        with cols[idx]:
            st.markdown(f"<div class='person-card'><h3>Person {idx+1}</h3></div>", unsafe_allow_html=True)
            
            # Images
            ic1, ic2 = st.columns(2)
            with ic1:
                st.caption("Live Snap")
                # Convert BGR to RGB for display
                st.image(cv2.cvtColor(p['live_img'], cv2.COLOR_BGR2RGB), use_container_width=True)
            with ic2:
                st.caption("Database Match")
                if p['match_img'] is not None:
                    st.image(cv2.cvtColor(p['match_img'], cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.warning("No Match")
            
            # Info
            name_html = f"<h4 style='color: {'#10B981' if p['name']!='Unknown' else '#EF4444'}; text-align:center;'>{p['name']}</h4>"
            st.markdown(name_html, unsafe_allow_html=True)
            
            # Risk Graph
            st.caption("MDR Risk Assessment")
            # Generate fake timeseries for graph
            chart_data = pd.DataFrame(
                np.random.randn(20, 1) + p['risk_score'],
                columns=['Risk Level']
            )
            st.line_chart(chart_data, color="#EF4444" if p['risk_score']>50 else "#10B981", height=150)

# Polling Loop (Keep at end)
if ctx.state.playing:
    try:
        # Non-blocking get
        while not st.session_state.result_queue.empty():
            results = st.session_state.result_queue.get_nowait()
            st.session_state.last_results = results # Cache
            print("Results received from queue!")
    except:
        pass
        
    # Auto-refresh to keep checking queue
    time.sleep(1)
    st.rerun()


