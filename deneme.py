import cv2 , supervision as sv , numpy as np , keyboard as kb
from ultralytics import YOLO

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

def main():
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("./yaprak.pt")
    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        global tracker_id , class_id
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]
        
        labels = [
            
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

        if kb.is_pressed("h"):
            tracker_id = 0
            class_id = 0
            


if __name__ == "__main__":
    main()