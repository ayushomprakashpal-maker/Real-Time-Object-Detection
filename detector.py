"""detector.py - YOLO detection + drawing. No database or UI code here."""
import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # downloads weights on first run
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

    def get_class_list(self):
        return sorted(self.class_names.values())

    def detect(self, frame, conf_threshold=0.7, selected_classes=None):
        """Run YOLO on one frame.

        Returns (annotated_frame, detections).
        Each detection is a dict: object_class, confidence, bbox_x/y/w/h.
        If selected_classes is empty/None, all classes are kept.
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        annotated = frame.copy()
        detections = []

        for box in results.boxes:
            name = self.class_names[int(box.cls[0])]
            if selected_classes and name not in selected_classes:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "object_class": name,
                "confidence": round(conf, 4),
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": x2 - x1,
                "bbox_h": y2 - y1,
            })

            # Draw bounding box + label
            label = f"{name} {conf:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated, detections
