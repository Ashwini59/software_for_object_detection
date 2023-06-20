from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("dnn_model/best.pt")
# path = 'imageattendance'

results = model.predict(source = "0", show = True)

print(results)
# if Key == 27:
#     break
