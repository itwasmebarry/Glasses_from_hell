import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO('yolov8n.pt') 

# 2. Start Camera
cap = cv2.VideoCapture(0)

# --- CONFIGURATION ---
healthy_ids = [46, 47, 48, 49, 50, 51] # Banana, Apple, Sandwich, Orange, Broccoli, Carrot
junk_ids = [52, 53, 54, 55]            # Hot Dog, Pizza, Donut, Cake
drink_ids = [39, 41]                   # Bottle, Cup

def get_junk_labels():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # --- LOGIC ---
                label = ""
                color = (255, 255, 255) # White default

                if class_id in healthy_ids:
                    label = f"HEALTHY: {model.names[class_id]}"
                    color = (0, 255, 0) # Green
                
                elif class_id in junk_ids:
                    label = f"JUNK: {model.names[class_id]}"
                    color = (0, 0, 255) # Red
                
                elif class_id in drink_ids:
                    label = f"DRINK: {model.names[class_id]}"
                    color = (255, 0, 0) # Blue

                # If it is a food/drink we recognize
                if label:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 1. Draw the Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # 2. Calculate Text Position (The Fix!)
                    # If the box is near the top (y1 < 30), draw text INSIDE the box
                    if y1 < 30:
                        text_y = y1 + 25 
                    else:
                        text_y = y1 - 10 

                    # 3. Draw the Text
                    # I changed the color to match the box so it's super visible
                    cv2.putText(frame, label, (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Food Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


get_junk_labels() 

cap.release()
cv2.destroyAllWindows()

