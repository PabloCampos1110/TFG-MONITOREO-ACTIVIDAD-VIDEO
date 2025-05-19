import csv

class MetricsLogger:
    def __init__(self, csv_filename="resultados_metrica.csv"):
        self.csv_filename = csv_filename
        self.data = []
        self.frame_id = 0

    def log(self, yolo_detections, orb_matches, fps=None):
        self.data.append({
            "Frame": self.frame_id,
            "YOLO_Detecciones": yolo_detections,
            "ORB_Coincidencias": orb_matches,
            "FPS": fps if fps is not None else 0
        })
        self.frame_id += 1

    def save(self):
        keys = ["Frame", "YOLO_Detecciones", "ORB_Coincidencias", "FPS"]
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)