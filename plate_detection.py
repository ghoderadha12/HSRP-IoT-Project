
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(
    r"runs\detect\runs\hsrp\plate_detector\weights\best.pt"
)

model = YOLO(str(MODEL_PATH))
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Detection started. Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    output = frame.copy()
    plate_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name != "Car-license-plates": 
            continue

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        plate_count += 1

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )

        label = f"PLATE {confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        print(
            f"{class_name}: {confidence:.2f}, "
            f"box=({x1},{y1},{x2},{y2})"
        )

    cv2.putText(
        output,
        f"Plates detected: {plate_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("HSRP Green Box Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
