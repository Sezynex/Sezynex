import cv2
import torch
from utils.general import non_max_suppression
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import scale_coords
from utils.plots import plot_one_box

def classify_objects(cam, model, device='cpu', conf_thres=0.25, iou_thres=0.45):
    while True:
        ret, img0 = cam.read()
        if not ret:
            break

        # Görüntüyü boyutlandırın ve renk dönüşümü yapın
        img = letterbox(img0, new_shape=640)[0]

        # PIL resmini PyTorch tensörüne dönüştürme
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # YOLOv8 modelini kullanarak nesne tespiti yapın
        pred = model(img, augment=False)[0]

        # Non-maximum suppression (NMS) uygula
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

        # Tespit edilen nesneleri sınıflandırıp sayın
        object_count = {}
        for det in pred:
            # Koordinatları ölçeklendirin
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Tespit edilen nesneleri sınıflandırın ve sayın
            for *xyxy, conf, cls in det:
                label = f"{int(cls)}"
                if label not in object_count:
                    object_count[label] = 1
                else:
                    object_count[label] += 1

                # Tespit edilen nesneleri çiz
                plot_one_box(xyxy, img0, label='Class', color=(0,255,0), line_thickness=3)

        # Tespit edilen nesneleri görselleştirin
        cv2.imshow('YOLOv8 Object Detection', img0)
        
        # Çıkış için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # USB kamera bağlantısını başlat
    cam = cv2.VideoCapture(0)

    # YOLOv8 modelini yükle
    model = attempt_load("yolov5x.pt", map_location=torch.device('cpu')).fuse().eval()

    # Nesne sınıflandırma ve sayımını yapın
    classify_objects(cam, model)
