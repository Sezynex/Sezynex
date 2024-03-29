import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

def main():
    line_counter = sv.LineZone(start=ZONE_POLYGON, end=ZONE_POLYGON)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("./yaprak.pt")
    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        
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
        print("class_id ", detections.),
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()
