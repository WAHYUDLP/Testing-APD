
import cv2
from ultralytics import YOLO

model = YOLO("best8.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    results = model(frame)
    frame = cv2.resize(frame, (416, 416))
    # frame = cv2.resize(frame, (640, 480))
    annotated_frame = results[0].plot()

    cv2.imshow("K3 Detection - Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break