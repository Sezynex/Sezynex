import cv2
import imutils
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

def main():
    cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Kamera bağlantısı başarısız")
        break
while True:
    results = model(frame, conf=0.5 save_crop=False)
    annotated_frame = results[0].plot()
    cv2.imshow("Usb cam", annotated_frame)

    if cv2.waitkey(1) & 0xFF == ord("q"):
        break
cam.release()
cv2.destroyALlWindows()