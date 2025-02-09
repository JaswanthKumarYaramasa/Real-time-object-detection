from fastapi import FastAPI, WebSocket
import cv2
import base64
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.websocket("/video")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection
        results = model(frame)[0]

        # Draw bounding boxes on the frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode()

        await websocket.send_text(jpg_as_text)

    cap.release()
