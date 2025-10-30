# detector.py
from ultralytics import YOLO
import cv2

class NoseDetector:
    def __init__(self, weights_path=None):
        self.model = YOLO(weights_path) if weights_path else None

    def crop_nose(self, img_bgr):
        # Se não há modelo, retorna a imagem inteira
        if self.model is None:
            return img_bgr
        h, w = img_bgr.shape[:2]
        results = self.model.predict(img_bgr, imgsz=640, conf=0.25, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return img_bgr
        # pega a caixa com maior confiança
        box = results[0].boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return img_bgr[y1:y2, x1:x2]
