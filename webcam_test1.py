
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam not found.")
    print("Try changing camera index 0 to 1.")
    raise SystemExit

print("Webcam working. Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    cv2.imshow("HSRP - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
