import torch
from PIL import Image
from pathlib import Path

# YOLOv5 kütüphanesini yükle
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def classify_objects(image_path, model, device='cpu', conf_thres=0.25, iou_thres=0.45):
    # Görüntüyü yükleyin
    img0 = Image.open(image_path)

    # Görüntüyü boyutlandırın ve renk dönüşümü yapın
    img = letterbox(img0, new_shape=640)[0]

    # PIL resmini PyTorch tensörüne dönüştürme
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # YOLOv5 modelini kullanarak nesne tespiti yapın
    pred = model(img, augment=False)[0]

    # Non-maximum suppression (NMS) uygula
    pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    # Tespit edilen nesneleri sınıflandırıp sayın
    object_count = {}
    for det in pred:
        # Koordinatları ölçeklendirin
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.size).round()

        # Tespit edilen nesneleri sınıflandırın ve sayın
        for *xyxy, conf, cls in det:
            label = f"{int(cls)}"
            if label not in object_count:
                object_count[label] = 1
            else:
                object_count[label] += 1

    return object_count

if __name__ == "__main__":
    # Model yüklemesi
    model = attempt_load("yolov5s.pt", map_location=torch.device('cpu')).fuse().eval()

    # Görüntü yolunu belirtin
    image_path = "test.jpg"

    # Nesne sınıflandırma ve sayımı yapın
    object_counts = classify_objects(image_path, model)

    # Sonuçları yazdır
    print("Tespit edilen nesneler:")
    for label, count in object_counts.items():
        print(f"Sınıf: {label}, Sayı: {count}")
