import os
# Suppress TensorFlow logs BEFORE importing it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import threading
import queue
import time
from PIL import Image

# ==========================================
# Configuration
# ==========================================
FACE_MODEL_PATH = 'weights/yolov8n-face-lindevs.pt'
HAND_MODEL_PATH = 'hand_yolov8n.pt'
KNOWN_DB_DIR = "people faces" # Using the directory name user mentioned/we saw
CONTACT_THRESHOLD = 250
COOLDOWN_SECONDS = 3

# ==========================================
# Application State
# ==========================================
if 'last_contact_time' not in st.session_state:
    st.session_state.last_contact_time = 0
if 'identified_person' not in st.session_state:
    st.session_state.identified_person = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'matched_image_path' not in st.session_state:
    st.session_state.matched_image_path = None

# Queue for communication between thread and main loop
result_queue = queue.Queue()

# ==========================================
# Caching Models
# ==========================================
@st.cache_resource
def load_models():
    # Helper to check/download if needed (omitted for speed, assuming they exist)
    face_model = YOLO(FACE_MODEL_PATH) if os.path.exists(FACE_MODEL_PATH) else YOLO('yolov8n.pt') # Fallback
    hand_model = YOLO(HAND_MODEL_PATH)
    return face_model, hand_model

try:
    model_face, model_hand = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'hand_yolov8n.pt' is in the directory.")
    st.stop()

# ==========================================
# Identification Thread
# ==========================================
def identify_worker(face_img):
    try:
        print(" [Debug] Starting identification...")
        st_time = time.time()
        
        # Facenet512 is generally faster and more accurate than VGG-Face for verification
        results = DeepFace.find(
            img_path=face_img, 
            db_path=KNOWN_DB_DIR, 
            model_name="Facenet512", 
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False, 
            silent=True
        )
        
        print(f" [Debug] Identification took: {time.time() - st_time:.2f}s")
        
        if len(results) > 0 and not results[0].empty:
            best_match = results[0].iloc[0]
            match_path = best_match['identity']
            distance = best_match['distance'] # Cosine distance
            
            print(f" [Debug] Match found: {match_path} (Dist: {distance:.4f})")
            
            # Extract Name
            path_parts = os.path.normpath(match_path).split(os.sep)
            
            # Check if it's inside a subfolder (people faces/Name/ing.jpg) or flat (people faces/Name.jpg)
            # path_parts[-1] is filename, path_parts[-2] is folder.
            
            # If the parent folder is the DB folder itself, then the name is the filename (minus extension)
            db_folder_name = os.path.basename(os.path.normpath(KNOWN_DB_DIR))
            parent_folder = path_parts[-2]
            
            if parent_folder == db_folder_name:
                # Flat structure case: "people faces/Aditya.jpg"
                name = os.path.splitext(path_parts[-1])[0]
            else:
                # Subfolder structure case: "people faces/Aditya/01.jpg"
                name = parent_folder

            result_queue.put({"name": name, "match_path": match_path, "status": "success"})
        else:
            print(" [Debug] No match found in database.")
            result_queue.put({"name": "Unknown", "match_path": None, "status": "no_match"})
            
    except Exception as e:
        print(f" [Error] Identification failed: {e}")
        result_queue.put({"status": "error", "message": str(e)})

# ==========================================
# UI Layout
# ==========================================
st.set_page_config(layout="wide", page_title="Handshake Recognition")

st.title("🤝 Handshake & Face Recognition System")

col_video, col_results = st.columns([2, 1])

with col_results:
    st.subheader("Identification Results")
    status_placeholder = st.empty()
    result_container = st.container()

    # Render previous results if any
    with result_container:
        if st.session_state.identified_person:
            st.success(f"Identified: **{st.session_state.identified_person}**")
            
            c1, c2 = st.columns(2)
            if st.session_state.captured_image is not None:
                c1.image(st.session_state.captured_image, caption="Captured Face", channels="BGR")
            
            if st.session_state.matched_image_path and os.path.exists(st.session_state.matched_image_path):
                # Load matched image safely
                matched_img = Image.open(st.session_state.matched_image_path)
                c2.image(matched_img, caption="Database Match")
        elif st.session_state.identified_person == "Unknown":
            st.warning("Person in contact, but not found in database.")
            if st.session_state.captured_image is not None:
                st.image(st.session_state.captured_image, caption="Captured Face", channels="BGR")

with col_video:
    st.header("Live Feed")
    run = st.checkbox('Run Camera', value=True)
    FRAME_WINDOW = st.image([])

    if run:
        # Use DirectShow backend on Windows to avoid MSMF errors
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from webcam.")
                break
            
            # YOLO Inference
            results_face = model_face(frame, verbose=False, conf=0.5)
            results_hand = model_hand(frame, verbose=False, conf=0.5)
            
            annotated_frame = frame.copy()
            
            # Draw Faces
            for box in results_face[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw Hands & Calculate Logic
            hand_centers = []
            for box in results_hand[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'Hand {conf:.2f}'
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                hand_centers.append((center_x, center_y))
            
            # Contact Logic
            contact_detected = False
            if len(hand_centers) >= 2:
                import math
                min_dist = float('inf')
                for i in range(len(hand_centers)):
                    for j in range(i + 1, len(hand_centers)):
                        h1 = hand_centers[i]
                        h2 = hand_centers[j]
                        dist = math.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
                        
                        if dist < min_dist: min_dist = dist
                        
                        if dist < CONTACT_THRESHOLD:
                            contact_detected = True
                            cv2.line(annotated_frame, h1, h2, (0, 0, 255), 4)
                            cv2.putText(annotated_frame, f"Person in Contact! {int(dist)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Trigger Identification on Contact
            if contact_detected:
                import time
                current_time = time.time()
                
                if current_time - st.session_state.last_contact_time > COOLDOWN_SECONDS:
                    st.session_state.last_contact_time = current_time
                    
                    # Find the largest/most prominent face to identify
                    # (Simple heuristic: largest area)
                    best_face_img = None
                    max_area = 0
                    
                    h, w, _ = frame.shape
                    
                    for fbox in results_face[0].boxes:
                        fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(w, fx2), min(h, fy2)
                        area = (fx2 - fx1) * (fy2 - fy1)
                        if area > max_area:
                            max_area = area
                            best_face_img = frame[fy1:fy2, fx1:fx2].copy()
                    
                    if best_face_img is not None and best_face_img.size > 0:
                        # Save to disk
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        save_dir = "contact_faces"
                        os.makedirs(save_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(save_dir, f"face_{timestamp}.jpg"), best_face_img)

                        status_placeholder.info("Identifying...")
                        st.session_state.captured_image = best_face_img # Store for UI
                        
                        # Start Thread
                        threading.Thread(target=identify_worker, args=(best_face_img,)).start()

            # Check for results from thread
            try:
                result = result_queue.get_nowait()
                if result['status'] == 'success':
                    st.session_state.identified_person = result['name']
                    st.session_state.matched_image_path = result['match_path']
                    # Trigger rerun to update UI in the other column immediately
                    st.rerun() 
                elif result['status'] == 'no_match':
                    st.session_state.identified_person = "Unknown"
                    st.session_state.matched_image_path = None
                    st.rerun()
                elif result['status'] == 'error':
                    st.error(f"Identification Error: {result.get('message')}")
            except queue.Empty:
                pass


            # Display Video
            FRAME_WINDOW.image(annotated_frame, channels='BGR')

        cap.release()
