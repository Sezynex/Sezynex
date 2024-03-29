import cv2
import argparse
import time
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
    reset_counter = 0
    reset_threshold = 10  # Sıfırlama eşiğini belirle (örneğin, 10)
    
    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True):
        
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detected_ids = "-".join(str(x) for x in result.boxes.id.cpu().numpy().astype(int))
            print("Algılanan nesnelerin idleri:", detected_ids)
        
        detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.4f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )

        # Her bir nesne için ID'yi kontrol et
        for _, _, _, tracker_id in detections:
            if tracker_id == 10:  # Eğer 10. ID'ye ulaşılırsa
                reset_counter += 1
                if reset_counter == reset_threshold:  # Sıfırlama eşiğine ulaşıldığında
                    detections.tracker_id = None
                    reset_counter = 0  # Sayaç sıfırlanır


        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
