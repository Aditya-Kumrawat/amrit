import os
import cv2
import time
from deepface import DeepFace

# ==========================================
# Configuration
# ==========================================
UNKNOWN_FACES_DIR = "contact_faces"
# Detected directory name from ls: "people faces"
KNOWN_DB_DIR = "people faces" 
MODEL_NAME = "VGG-Face"

print("--- Starting Identification System ---")
print(f"Monitoring folder: {UNKNOWN_FACES_DIR}")
print(f"Database folder: {KNOWN_DB_DIR}")

if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

# Maintain a set of processed files to avoid re-processing
processed_files = set()

while True:
    # Get current files
    current_files = set([f for f in os.listdir(UNKNOWN_FACES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Determine new files
    new_files = current_files - processed_files
    
    if new_files:
        print(f"\nFound {len(new_files)} new face(s)...")
        
        for file in new_files:
            img_path = os.path.join(UNKNOWN_FACES_DIR, file)
            print(f"Analyzing: {file}...")
            
            try:
                # Run DeepFace Match
                # enforce_detection=False because it's already a crop
                results = DeepFace.find(
                    img_path=img_path, 
                    db_path=KNOWN_DB_DIR, 
                    model_name=MODEL_NAME, 
                    detector_backend="opencv",
                    enforce_detection=False, 
                    silent=True
                )

                if len(results) > 0 and not results[0].empty:
                    best_match = results[0].iloc[0]
                    match_path = best_match['identity']
                    
                    # Extract Name
                    # Assuming structure: people faces/Name/image.jpg
                    path_parts = os.path.normpath(match_path).split(os.sep)
                    # Name is folder name
                    if len(path_parts) >= 2:
                        person_name = path_parts[-2]
                    else:
                        person_name = "Unknown"

                    print(f"✅ IDENTIFIED: {person_name}")
                    print(f"   (Matched with: {match_path})")
                else:
                    print("❌ Unknown Person (No match in database)")
                    
            except Exception as e:
                print(f"   Error: {e}")
            
            # Mark as processed
            processed_files.add(file)
            
    time.sleep(1) # Check every second
