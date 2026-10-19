


import cv2
import easyocr
from ultralytics import YOLO

from pathlib import Path

MODEL_PATH = Path(
    r"C:\Users\Sanjay Ghode\OneDrive\Documents\Project HSRP\runs\detect\runs\hsrp\plate_detector\weights\best.pt"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

model = YOLO(str(MODEL_PATH))
print("SUCCESS: YOLO model loaded!")
print("Model classes:", model.names)
reader = easyocr.Reader(["en"], gpu=False)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam failed to open.")
    cap.release()
    raise SystemExit

print("SUCCESS: Webcam opened!")

print("License plate detection and OCR started.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.15,
        verbose=False
    )

    for box in results[0].boxes:
        confidence = float(box.conf[0])

        # Get plate coordinates
        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        # Keep coordinates inside the image
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop the detected number plate
        plate_crop = frame[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Read characters using OCR
        ocr_results = reader.readtext(
            plate_crop,
            detail=0
        )
        plate_text = " ".join(ocr_results).strip()

        if not plate_text:
            plate_text = "Text not recognized"

        # Draw a green rectangle around the plate
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        # Display recognized text and confidence
        label = f"{plate_text} ({confidence:.2f})"
        label_y = max(y1 - 10, 25)

        cv2.putText(
            frame, label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        print(
            f"Detection confidence: {confidence:.2f} | "
            f"Recognized text: {plate_text}"
        )

    cv2.imshow("HSRP Detection and OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()