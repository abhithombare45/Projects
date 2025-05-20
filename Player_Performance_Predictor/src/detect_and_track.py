from ultralytics import YOLO
import cv2
import os
import pandas as pd

# Load YOLO model
model = YOLO('../models/yolov8n.pt')

# Directory of frames
frames_dir = '../data/processed/frames'
frames = sorted(os.listdir(frames_dir))

# Store tracking data
tracking_data = []

# Simple centroid tracking (for demo purposes)
player_positions = {}

for frame_file in frames:
    frame_path = os.path.join(frames_dir, frame_file)
    frame = cv2.imread(frame_path)
    
    # Detect players and ball
    results = model(frame)
    frame_data = {'frame': frame_file}
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls in [0, 32]:  # 0: person (player), 32: sports ball
                x, y, w, h = box.xywh[0]
                centroid = (int(x), int(y))
                label = 'player' if cls == 0 else 'ball'
                
                # Track players using centroid
                if label == 'player':
                    player_id = f'player_{len(player_positions) + 1}'
                    player_positions[player_id] = centroid
                    frame_data[player_id] = centroid
                
                frame_data[label] = centroid
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x - w/2), int(y - h/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    tracking_data.append(frame_data)
    cv2.imwrite(frame_path, frame)  # Save annotated frame

# Save tracking data
df = pd.DataFrame(tracking_data)
df.to_csv('../data/processed/tracking_data.csv', index=False)
print("Tracking complete")
