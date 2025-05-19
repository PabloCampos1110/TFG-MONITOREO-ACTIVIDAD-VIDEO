from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu'):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        results = self.model.predict(frame, imgsz=640, device=self.device, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        return detections, confidences, class_ids




