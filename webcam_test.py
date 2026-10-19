
import cv2

# Open the default laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open the webcam.")
    raise SystemExit(1)

print("Webcam started. Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read a camera frame.")
        break

    cv2.imshow("HSRP Detection - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
cv2.imshow("HSRP Detection - Webcam Test", frame)
