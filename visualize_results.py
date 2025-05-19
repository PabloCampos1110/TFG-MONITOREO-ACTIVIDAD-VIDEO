import cv2

def draw_detections(frame, detections, confidences, class_ids, threshold=0.5):
    for (bbox, score, cls_id) in zip(detections, confidences, class_ids):
        if score > threshold:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID {int(cls_id)} {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return frame