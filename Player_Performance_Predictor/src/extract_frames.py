import cv2
import os

# Create output directory for frames
os.makedirs('../data/processed/frames', exist_ok=True)

# Load video
video_path = '../data/raw/soccer_match.mp4'
cap = cv2.VideoCapture(video_path)

# Extract frames
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Resize to 720p if needed
    frame = cv2.resize(frame, (1280, 720))
    cv2.imwrite(f'../data/processed/frames/frame_{frame_count:04d}.jpg', frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames")
