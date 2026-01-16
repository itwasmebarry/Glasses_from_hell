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

# --- CALORIE DATABASE (The Engineering Estimates) ---
# Units are roughly "Per Serving" or "Per Item"
calorie_map = {
    # Healthy
    46: 105, # Banana
    47: 95,  # Apple
    48: 300, # Sandwich (Average)
    49: 45,  # Orange
    50: 31,  # Broccoli (1 cup)
    51: 25,  # Carrot
    # Junk
    52: 290, # Hot Dog
    53: 285, # Pizza (Per Slice estimate)
    54: 195, # Donut
    55: 260, # Cake (Slice)
    39: 150, # Bottle (assuming sugary drink)
    41: 140, # Cup
}

# --- PROCEDURAL ASSETS (Roaches & Halos) ---
def draw_roach(img, x, y, angle):
    color = (20, 35, 70) 
    body_len, head_len = 15, 5
    
    # Legs
    for i in range(-1, 2):
        lx1, ly1 = int(x + i*5), int(y + 5)
        lx2, ly2 = int(x + (i*8) - 15), int(y + 15)
        cv2.line(img, (lx1, ly1), (lx2, ly2), color, 1)
        rx1, ry1 = int(x + i*5), int(y - 5)
        rx2, ry2 = int(x + (i*8) - 15), int(y - 15)
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, 1)

    # Body & Head
    cv2.ellipse(img, (x, y), (body_len, 7), angle, 0, 360, color, -1)
    rad = np.deg2rad(angle)
    hx, hy = int(x + (body_len-2)*np.cos(rad)), int(y + (body_len-2)*np.sin(rad))
    cv2.circle(img, (hx, hy), head_len, color, -1)

    # Antennae
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
            r[3] += random.uniform(-2, 2)
            r[4] += random.uniform(-2, 2)
            r[3] *= 0.9; r[4] *= 0.9
            r[0] += int(r[3]); r[1] += int(r[4])
            r[0] = np.clip(r[0], x1, x2)
            r[1] = np.clip(r[1], y1, y2)
            if abs(r[3]) > 0.1 or abs(r[4]) > 0.1:
                r[2] = np.degrees(np.arctan2(r[4], r[3]))
            draw_roach(frame, int(r[0]), int(r[1]), int(r[2]))

# --- MAIN APP ---
model = YOLO('yolov8n.pt') 

pipeline = None
if REALSENSE_AVAILABLE:
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        print("RealSense camera initialized successfully")
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        print("Falling back to regular webcam...")
        pipeline = None
else:
    print("RealSense not available, using regular webcam...")

healthy_ids = [46, 47, 48, 49, 50, 51] 
junk_ids = [52, 53, 54, 55, 39, 41] 
swarms = {}

# Load idiot sandwich video once at startup
try:
    idiot_sandwich_video = cv2.VideoCapture("audio/idiot_sandwich.mp4")
    if not idiot_sandwich_video.isOpened():
        idiot_sandwich_video = None
        print("Warning: Could not load idiot_sandwich.mp4")
except Exception as e:
    idiot_sandwich_video = None
    print(f"Warning: Error loading video: {e}")


# Load idiot sandwich video once at startup
try:
    justdoit_video = cv2.VideoCapture("audio/justdoit.mp4")
    if not justdoit_video.isOpened():
        justdoit_video = None
        print("Warning: Could not load justdoit.mp4")
except Exception as e:
    justdoit_video = None
    print(f"Warning: Error loading video: {e}")


if pipeline is None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
else:
    cap = None

# Initialize audio once at startup
audio_initialized = False
if PYGAME_AVAILABLE:
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("audio/idiot_sanwhich_audio.mp3")
        audio_initialized = True
    except Exception as e:
        print(f"Audio initialization error: {e}")
        audio_initialized = False

while True:
    if pipeline is not None:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
    else:
        success, frame = cap.read()
        if not success:
            break
    
    clean_frame = frame.copy()
    
    # Variables to calculate total screen calories
    total_screen_cals = 0
    junk_food_detected = False
    healthy_food_detected = False
    
    # Read video frame once per loop
    overlay_frame = None
    if idiot_sandwich_video is not None and idiot_sandwich_video.isOpened():
        ret, overlay_frame = idiot_sandwich_video.read()
        if not ret:
            idiot_sandwich_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, overlay_frame = idiot_sandwich_video.read()

    overlay_frame_just_do_it = None
    if justdoit_video is not None and justdoit_video.isOpened():
        ret, overlay_frame_just_do_it = justdoit_video.read()
        if not ret:
            justdoit_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, overlay_frame_just_do_it = justdoit_video.read()

    results = model.track(frame, persist=True, verbose=False)
    #print(results)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            
            # GET CALORIES
            cals = calorie_map.get(cls, 0) # Default to 0 if unknown
            total_screen_cals += cals
            cal_text = f"{cals} kcal"

            # --- JUNK FOOD LOGIC ---
            if cls in junk_ids:
                junk_food_detected = True
                roi = frame[y1:y2, x1:x2]
                #roi[:, :, 2] = 0 # Kill Red
                frame[y1:y2, x1:x2] = roi 

                if track_id not in swarms: swarms[track_id] = RoachSwarm()
                swarms[track_id].update(x1, y1, x2, y2, frame)
                
                # Draw Warning Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw Calorie Label (Red Background)
                cv2.rectangle(frame, (x1, y1-30), (x1+120, y1), (0, 0, 255), -1)
                cv2.putText(frame, cal_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # adding the video of idiot sandwich mp4 onto the overlay 
                if overlay_frame is not None:
                    box_width, box_height = x2 - x1, y2 - y1
                    if box_width > 0 and box_height > 0:
                        roi = frame[y1:y2, x1:x2]
                        resized_overlay = cv2.resize(overlay_frame, (roi.shape[1], roi.shape[0]))
                        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, resized_overlay, 0.7, 0)
                        
                        if audio_initialized and pygame.mixer.music.get_busy() == 0:
                            pygame.mixer.music.play(-1)  # -1 for loop


            # --- HEALTHY FOOD LOGIC ---
            elif cls in healthy_ids:
                # Beautify
                roi = clean_frame[y1:y2, x1:x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype("float32")
                h, s, v = cv2.split(hsv)
                s, v = s*1.5, v*1.2
                hsv = cv2.merge([h, np.clip(s, 0, 255), np.clip(v, 0, 255)])
                enhanced = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
                frame[y1:y2, x1:x2] = cv2.addWeighted(enhanced, 0.7, cv2.GaussianBlur(enhanced,(21,21),0), 0.5, 0)
                healthy_food_detected = True
                # Halo
                cx = int((x1+x2)/2)
                cv2.ellipse(frame, (cx, y1-30), (int((x2-x1)/2), 10), 0, 0, 360, (150, 255, 255), 4) 
                
                # Draw Calorie Label (Gold Background)
                cv2.rectangle(frame, (x1, y1-30), (x1+120, y1), (0, 215, 255), -1)
                cv2.putText(frame, cal_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

                #adding overlay for healthy food
                if overlay_frame_just_do_it is not None:
                    box_width, box_height = x2 - x1, y2 - y1
                    if box_width > 0 and box_height > 0:
                        roi = frame[y1:y2, x1:x2]
                        resized_overlay = cv2.resize(overlay_frame_just_do_it, (roi.shape[1], roi.shape[0]))
                        frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.3, resized_overlay, 0.7, 0)
                        
                        if audio_initialized and pygame.mixer.music.get_busy() == 0:
                            pygame.mixer.music.play(-1)  # -1 for loop

    # Stop audio if no junk food is detected
    if audio_initialized and not junk_food_detected and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # Stop audio if no healthy food is detected
    if audio_initialized and not healthy_food_detected and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    # --- TOTAL CALORIE DASHBOARD ---
    # Display the total detected calories in the top left
    cv2.rectangle(frame, (10, 10), (300, 60), (50, 50, 50), -1) # Dark bg
    cv2.putText(frame, f"TOTAL: {total_screen_cals} kcal", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Calorie Roach Cam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if pipeline is not None:
    pipeline.stop()
if cap is not None:
    cap.release()
if idiot_sandwich_video is not None:
    idiot_sandwich_video.release()
if justdoit_video is not None:
    justdoit_video.release()
cv2.destroyAllWindows()
