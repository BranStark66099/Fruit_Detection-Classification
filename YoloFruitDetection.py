import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

def exit_program():
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)


    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
        exit_program()
        break
