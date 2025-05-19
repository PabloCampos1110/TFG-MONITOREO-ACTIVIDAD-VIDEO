import cv2
from load_video import load_video
from detect_activity import YOLOv8Detector
from orb_analysis import ORBTracker
from visualize_results import draw_detections
from metrics_logger import MetricsLogger

# Configuración inicial
video_path = 'videos/multiples2.mp4'
save_path = 'output_resultado.avi'
device = 'cpu'

# Inicialización de componentes
cap = load_video(video_path)
detector = YOLOv8Detector(device=device)
orb_tracker = ORBTracker()
logger = MetricsLogger("metricas_sistema.csv")

# Parámetros de salida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
prev_descriptors = None

# Bucle de procesamiento frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ────── PREPROCESAMIENTO ──────
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Mejora de contraste (útil en baja luz)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    # ────── DETECCIÓN CON YOLOV8 ──────
    detections, confidences, class_ids = detector.detect(frame)

    # ────── ANÁLISIS ORB ──────
    keypoints, descriptors = orb_tracker.get_keypoints_descriptors(frame)

    num_matches = 0
    if prev_descriptors is not None and descriptors is not None:
        matches = orb_tracker.match_descriptors(prev_descriptors, descriptors)
        num_matches = len(matches)

    prev_descriptors = descriptors

    # ────── VISUALIZACIÓN ──────
    frame = draw_detections(frame, detections, confidences, class_ids)
    cv2.putText(frame, f"Matches ORB: {num_matches}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)  # Guarda en el vídeo
    logger.log(len(detections), num_matches)  # Guarda métricas

    cv2.imshow('Actividad detectada', frame)  # Muestra en tiempo real

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cierre y limpieza
cap.release()
out.release()
cv2.destroyAllWindows()
logger.save()