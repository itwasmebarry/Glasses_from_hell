import cv2
import numpy as np
import random
from ultralytics import YOLO

# --- ENGINEERING ASSETS: PROCEDURAL DRAWING FUNCTIONS ---

def draw_roach(img, x, y, angle):
    """
    Draws a procedural cockroach at (x,y) with a specific rotation.
    No external image files required. Pure OpenCV geometry.
    """
    # Create a local coordinate system for the roach
    # Body color: Dark Brown
    color = (20, 35, 70) 
    
    # We rotate the drawing canvas manually by calculating offsets
    # Length of roach parts
    body_len = 15
    head_len = 5
    
    # 1. Draw Legs (6 legs) - Simple lines jittering
    for i in range(-1, 2):
        # Left legs
        lx1 = int(x + i * 5)
        ly1 = int(y + 5)
        lx2 = int(x + (i * 8) - 15)
        ly2 = int(y + 15)
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, 1)
        
        # Right legs
        rx1 = int(x + i * 5)
        ry1 = int(y - 5)
        rx2 = int(x + (i * 8) - 15)
        ry2 = int(y - 15)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, 1)

    # 2. Draw Body (Ellipse)
    cv2.ellipse(img, (x, y), (body_len, 7), angle, 0, 360, color, -1)
    
    # 3. Draw Head (Small Ellipse offset)
    # Calculate head position based on angle
    rad = np.deg2rad(angle)
    hx = int(x + (body_len - 2) * np.cos(rad))
    hy = int(y + (body_len - 2) * np.sin(rad))
    cv2.circle(img, (hx, hy), head_len, color, -1)

    # 4. Antennae (Long thin lines from head)
    ant_len = 25
    # Left antenna
    ax1 = int(hx + ant_len * np.cos(rad - 0.5))
    ay1 = int(hy + ant_len * np.sin(rad - 0.5))
    cv2.line(img, (hx, hy), (ax1, ay1), color, 1)
    # Right antenna
    ax2 = int(hx + ant_len * np.cos(rad + 0.5))
    ay2 = int(hy + ant_len * np.sin(rad + 0.5))
    cv2.line(img, (hx, hy), (ax2, ay2), color, 1)

class RoachSwarm:
    """Manages a swarm of roaches for a specific target box"""
    def __init__(self):
        self.roaches = [] # List of [x, y, angle, speed_x, speed_y]

    def update(self, x1, y1, x2, y2, frame):
        # Spawn roaches if too few (cap at 8 per item)
        if len(self.roaches) < 8:
            # Spawn at random edge of box
            rx = random.randint(x1, x2)
            ry = random.randint(y1, y2)
            self.roaches.append([rx, ry, random.randint(0, 360), 0, 0])

        # Move roaches
        for r in self.roaches:
            # Jitter movement
            r[3] += random.uniform(-2, 2) # Change velocity X
            r[4] += random.uniform(-2, 2) # Change velocity Y
            
            # Dampen velocity (friction)
            r[3] *= 0.9
            r[4] *= 0.9
            
            # Update Position
            r[0] += int(r[3])
            r[1] += int(r[4])
            
            # Keep inside the box (roughly)
            r[0] = np.clip(r[0], x1, x2)
            r[1] = np.clip(r[1], y1, y2)
            
            # Orient head towards movement
            if abs(r[3]) > 0.1 or abs(r[4]) > 0.1:
                r[2] = np.degrees(np.arctan2(r[4], r[3]))

            # Draw
            draw_roach(frame, int(r[0]), int(r[1]), int(r[2]))

# --- MAIN APPLICATION ---

# 1. Load Model
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)

# 2. Config
healthy_ids = [46, 47, 48, 49, 50, 51] 
junk_ids = [52, 53, 54, 55, 41, 39] # Added drinks to junk for testing
swarms = {} # Dictionary to store a swarm for each tracked object ID

print("System Active. Show food.")

while True:
    success, frame = cap.read()
    if not success: break
    
    # Copy frame for "Divine Blur" calculations
    clean_frame = frame.copy()

    # Track=True gives us stable IDs (so the roach remembers which pizza it's on)
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            
            # --- JUNK FOOD (ROACH SWARM) ---
            if cls in junk_ids:
                # 1. The Mold Filter (Cold/Dead colors)
                roi = frame[y1:y2, x1:x2]
                roi[:, :, 2] = 0 # Kill Red channel
                frame[y1:y2, x1:x2] = roi # Paste back

                # 2. The Animation System
                # Initialize a swarm for this specific object ID if not exists
                if track_id not in swarms:
                    swarms[track_id] = RoachSwarm()
                
                # Update and Draw Roaches
                swarms[track_id].update(x1, y1, x2, y2, frame)
                
                # Label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "INFESTED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # --- HEALTHY FOOD (DIVINE BEAUTIFICATION) ---
            elif cls in healthy_ids:
                # 1. Saturation Boost
                roi = clean_frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype("float32")
                (h, s, v) = cv2.split(hsv)
                s = s * 1.5 # Boost saturation by 50%
                v = v * 1.2 # Boost brightness by 20%
                s = np.clip(s, 0, 255)
                v = np.clip(v, 0, 255)
                hsv = cv2.merge([h, s, v])
                enhanced_roi = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
                
                # 2. Soft Bloom/Glow
                blur = cv2.GaussianBlur(enhanced_roi, (21, 21), 0)
                enhanced_roi = cv2.addWeighted(enhanced_roi, 0.6, blur, 0.4, 0)
                
                # Paste the beautiful fruit back
                frame[y1:y2, x1:x2] = enhanced_roi

                # 3. Procedural Halo
                center_x = int((x1 + x2) / 2)
                # Draw Halo Ellipse floating above
                cv2.ellipse(frame, (center_x, y1 - 30), (int((x2-x1)/2), 10), 
                            0, 0, 360, (0, 255, 255), 2) # Yellow ring
                # Add "Glow" to the halo (draw it again thicker and blurrier)
                cv2.ellipse(frame, (center_x, y1 - 30), (int((x2-x1)/2), 10), 
                            0, 0, 360, (150, 255, 255), 6) 
                
                cv2.putText(frame, "DIVINE", (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("The Judge", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()