import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path='yolo11n.pt', cam_index=0, conf_threshold=0.4):
        self.yolo = YOLO(model_path)
        self.videoCap = cv2.VideoCapture(0)
        self.conf_threshold = conf_threshold
        self.base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]

    def get_colour(self, cls_num):
        color_index = cls_num % len(self.base_colors)
        color = [
            (self.base_colors[color_index][i] +
             self.increments[color_index][i] * (cls_num // len(self.base_colors))) % 256
            for i in range(3)
        ]
        return tuple(color)

    def process_frame(self, frame):
        results = self.yolo.track(frame, stream=True)
        for result in results:
            class_names = result.names
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf > self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    colour = self.get_colour(cls)
                    label = f"{class_names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        return frame

    def run(self):
        while True:
            ret, frame = self.videoCap.read()
            if not ret:
                continue

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Process and draw detections
            
            frame = self.process_frame(frame)

            # Display result
            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.videoCap.release()
        cv2.destroyAllWindows()


# Instantiate and run
if __name__ == "__main__":


    detector = ObjectDetector()
    detector.run()