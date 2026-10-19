
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(
    r"D:\HSRP_Project\runs\hsrp\plate_detector\weights\best.pt"
)

# 1. Check the model file
if not MODEL_PATH.exists():
    print("ERROR: best.pt not found!")
    print("Expected location:", MODEL_PATH.resolve())
    raise SystemExit

# 2. Load YOLO
model = YOLO(str(MODEL_PATH))
print("SUCCESS: YOLO model loaded!")
print("Classes:", model.names)

# 3. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not open.")
    raise SystemExit

print("SUCCESS: Webcam opened!")
print("Press Q to quit.")

# 4. Run live detection
while True:
    success, frame = cap.read()

    if not success:
        print("ERROR: Cannot read webcam frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.15,
        verbose=False
    )

    # Draw YOLO detections
    annotated = results[0].plot()

    cv2.imshow("YOLO Webcam Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
