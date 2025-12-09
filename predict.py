import argparse
from ultralytics import YOLO
from PIL import Image
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov8n-face-lindevs.pt', help='Face weights path')
    parser.add_argument('--weights-hand', default='hand_yolov8n.pt', help='Hand weights path')
    parser.add_argument('--source', default='data/images/bus.jpg')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.75, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/yolov8n-face-lindevs.pt', help='Face weights path')
    parser.add_argument('--weights-hand', default='hand_yolov8n.pt', help='Hand weights path')
    parser.add_argument('--source', default='data/images/bus.jpg', help='Image or 0 for webcam')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.5, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0', help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    # Load models
    model_face = YOLO(opt.weights)
    model_hand = YOLO(opt.weights_hand)

    source = opt.source
    is_webcam = source.isnumeric() or source.endswith('.txt') or (source.lower() == '0')

    if is_webcam:
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results_face = model_face(frame, verbose=False, conf=opt.conf)
            results_hand = model_hand(frame, verbose=False, conf=opt.conf)
            
            # Plot Face
            annotated_frame = results_face[0].plot()
            
            # Plot Hands and Handshake Logic
            boxes_hand = results_hand[0].boxes
            hand_centers = []
            
            for box in boxes_hand:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f'{model_hand.names[cls]} {conf:.2f}'
                
                # Draw hand
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate center
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                hand_centers.append((center_x, center_y))
            
            # Handshake Detection (Simple distance check)
            cv2.putText(annotated_frame, f"Hands: {len(hand_centers)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if len(hand_centers) >= 2:
                # Check distance between all pairs (simplified to first two closest or just iterate)
                import math
                min_dist = float('inf')
                for i in range(len(hand_centers)):
                    for j in range(i + 1, len(hand_centers)):
                        h1 = hand_centers[i]
                        h2 = hand_centers[j]
                        dist = math.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                        
                        # Threshold for handshake (adjustable)
                        # Increased to 250 for easier detection
                        if dist < 250: 
                            cv2.putText(annotated_frame, f"Person in Contact! Dist: {int(dist)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            # Draw line between hands
                            cv2.line(annotated_frame, h1, h2, (0, 0, 255), 4)
                            
                            # Capture Faces Logic (with cooldown)
                            import time
                            import os
                            
                            current_time = time.time()
                            # Initialize cooldown tracker if not exists (using a distinct variable name to avoid scope issues if this block runs repeatedly)
                            if 'last_save_time' not in globals():
                                global last_save_time
                                last_save_time = 0
                                
                            if current_time - last_save_time > 2.0: # 2 seconds cooldown
                                last_save_time = current_time
                                
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                # Flat structure for easier monitoring
                                save_dir = "contact_faces"
                                os.makedirs(save_dir, exist_ok=True)
                                
                                # Iterate over faces and save
                                for idx, fbox in enumerate(results_face[0].boxes):
                                    fx1, fy1, fx2, fy2 = map(int, fbox.xyxy[0])
                                    h, w, _ = frame.shape
                                    fx1, fy1 = max(0, fx1), max(0, fy1)
                                    fx2, fy2 = min(w, fx2), min(h, fy2)
                                    
                                    face_img = frame[fy1:fy2, fx1:fx2]
                                    if face_img.size > 0:
                                        # Filename details
                                        filename = f"face_{timestamp}_{idx}.jpg"
                                        cv2.imwrite(os.path.join(save_dir, filename), face_img)
                                
                                cv2.putText(annotated_frame, "FACES CAPTURED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        else:
                            # Draw connecting line to show we are checking
                            cv2.line(annotated_frame, h1, h2, (255, 255, 0), 2)
                
                if min_dist != float('inf'):
                     cv2.putText(annotated_frame, f"Min Dist: {int(min_dist)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow('YOLOv8 Face + Hand Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Static Image
        img = Image.open(source) if not isinstance(source, str) else Image.open(source) 
        # Actually YOLO load handles str path well, but we need cv2 image for consistency or just let YOLO load it
        results_face = model_face(source, verbose=False) # YOLO loads source
        results_hand = model_hand(source, verbose=False)
        
        annotated_frame = results_face[0].plot() # BGR numpy array
        
        boxes_hand = results_hand[0].boxes
        hand_centers = []
        
        for box in boxes_hand:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f'{model_hand.names[cls]} {conf:.2f}'
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            hand_centers.append((center_x, center_y))
            
        if len(hand_centers) >= 2:
                import math
                for i in range(len(hand_centers)):
                    for j in range(i + 1, len(hand_centers)):
                        h1 = hand_centers[i]
                        h2 = hand_centers[j]
                        dist = math.sqrt((h1[0] - h2[0])**2 + (h1[1] - h2[1])**2)
                        
                        if dist < 100: 
                            cv2.putText(annotated_frame, "HANDSHAKE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            cv2.line(annotated_frame, h1, h2, (0, 0, 255), 2)

        img_show = Image.fromarray(annotated_frame[..., ::-1])
        img_show.show()
