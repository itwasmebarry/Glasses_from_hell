import cv2
import numpy as np
import random
from ultralytics import YOLO

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("pyrealsense2 not available, will use regular webcam")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available, audio will be disabled")

# --- CALORIE DATABASE ---
calorie_map = {
    46: 105, 47: 95, 48: 300, 49: 45, 50: 31, 51: 25, # Healthy
    52: 290, 53: 285, 54: 195, 55: 260, 39: 150, 41: 140 # Junk
}

# --- PROCEDURAL ASSETS (Roaches) ---
def draw_roach(img, x, y, angle):
    color = (20, 35, 70) 
    body_len, head_len = 15, 5
    for i in range(-1, 2):
        lx1, ly1 = int(x + i*5), int(y + 5)
        lx2, ly2 = int(x + (i*8) - 15), int(y + 15)
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, 1)
        rx1, ry1 = int(x + i*5), int(y - 5)
        rx2, ry2 = int(x + (i*8) - 15), int(y - 15)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, 1)
    cv2.ellipse(img, (x, y), (body_len, 7), angle, 0, 360, color, -1)
    rad = np.deg2rad(angle)
    hx, hy = int(x + (body_len-2)*np.cos(rad)), int(y + (body_len-2)*np.sin(rad))
    cv2.circle(img, (hx, hy), head_len, color, -1)
    ant_len = 25
    ax1, ay1 = int(hx + ant_len*np.cos(rad-0.5)), int(hy + ant_len*np.sin(rad-0.5))
    ax2, ay2 = int(hx + ant_len*np.cos(rad+0.5)), int(hy + ant_len*np.sin(rad+0.5))
    cv2.line(img, (hx, hy), (ax1, ay1), color, 1)
    cv2.line(img, (hx, hy), (ax2, ay2), color, 1)

class RoachSwarm:
    def __init__(self):
        self.roaches = [] 
    def update(self, x1, y1, x2, y2, frame):
        if len(self.roaches) < 8:
            rx, ry = random.randint(x1, x2), random.randint(y1, y2)
            self.roaches.append([rx, ry, random.randint(0, 360), 0, 0])
        for r in self.roaches:
            r[3] += random.uniform(-2, 2); r[4] += random.uniform(-2, 2)
            r[3] *= 0.9; r[4] *= 0.9
            r[0] += int(r[3]); r[1] += int(r[4])
            r[0] = np.clip(r[0], x1, x2); r[1] = np.clip(r[1], y1, y2)
            if abs(r[3]) > 0.1 or abs(r[4]) > 0.1:
                r[2] = np.degrees(np.arctan2(r[4], r[3]))
            draw_roach(frame, int(r[0]), int(r[1]), int(r[2]))

# --- INITIALIZATION ---
model = YOLO('yolov8n.pt') 
pipeline = None
if REALSENSE_AVAILABLE:
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
    except: pipeline = None

if pipeline is None:
    cap = cv2.VideoCapture(0)
else: cap = None

healthy_ids = [46, 47, 48, 49, 50, 51] 
junk_ids = [52, 53, 54, 55, 39, 41] 
swarms = {}

# Load Videos
idiot_sandwich_video = cv2.VideoCapture("audio/idiot_sandwich.mp4")
justdoit_video = cv2.VideoCapture("audio/justdoit.mp4")

# --- FIXED AUDIO INITIALIZATION ---
audio_initialized = False
sound_idiot = None
sound_justdoit = None

if PYGAME_AVAILABLE:
    try:
        pygame.mixer.init()
        # Use Sound objects instead of music.load
        sound_idiot = pygame.mixer.Sound("audio/idiot_sandwich_audio.mp3")
        sound_justdoit = pygame.mixer.Sound("audio/justdoit.mp3")
        audio_initialized = True
    except Exception as e:
        print(f"Audio Error: {e}")

while True:
    if pipeline is not None:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        frame = np.asanyarray(color_frame.get_data())
    else:
        success, frame = cap.read()
        if not success: break
    
    clean_frame = frame.copy()
    total_screen_cals = 0
    junk_food_detected = False
    healthy_food_detected = False
    
    # Video Overlays
    ret1, overlay_idiot = idiot_sandwich_video.read()
    if not ret1:
        idiot_sandwich_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, overlay_idiot = idiot_sandwich_video.read()

    ret2, overlay_justdoit = justdoit_video.read()
    if not ret2:
        justdoit_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, overlay_justdoit = justdoit_video.read()

    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cals = calorie_map.get(cls, 0)
            total_screen_cals += cals
            
            # JUNK FOOD
            if cls in junk_ids:
                junk_food_detected = True
                if track_id not in swarms: swarms[track_id] = RoachSwarm()
                swarms[track_id].update(x1, y1, x2, y2, frame)
                
                if overlay_idiot is not None:
                    roi = frame[y1:y2, x1:x2]
                    resized = cv2.resize(overlay_idiot, (roi.shape[1], roi.shape[0]))
                    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, resized, 0.7, 0)

            # HEALTHY FOOD
            elif cls in healthy_ids:
                healthy_food_detected = True
                if overlay_justdoit is not None:
                    roi = frame[y1:y2, x1:x2]
                    resized = cv2.resize(overlay_justdoit, (roi.shape[1], roi.shape[0]))
                    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, resized, 0.7, 0)

    # --- FIXED AUDIO LOGIC ---
    if audio_initialized:
        # Channel 0 for Junk Food (Idiot Sandwich)
        if junk_food_detected:
            if not pygame.mixer.Channel(0).get_busy():
                pygame.mixer.Channel(0).play(sound_idiot, loops=-1)
        else:
            pygame.mixer.Channel(0).stop()

        # Channel 1 for Healthy Food (Just Do It)
        if healthy_food_detected:
            if not pygame.mixer.Channel(1).get_busy():
                pygame.mixer.Channel(1).play(sound_justdoit, loops=-1)
        else:
            pygame.mixer.Channel(1).stop()

    cv2.imshow("Calorie Roach Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Cleanup
if pipeline: pipeline.stop()
if cap: cap.release()
idiot_sandwich_video.release()
justdoit_video.release()
cv2.destroyAllWindows()