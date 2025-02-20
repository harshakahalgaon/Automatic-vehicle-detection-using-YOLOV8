from ultralytics import YOLO
import cv2

model = YOLO('D:\\final_yolo_custom_model\\best.pt')
#results = model("D:\\PROJECTS\\yolo_obj_det\\Chapter 5 - Running Yolo\\Images\\2.png", show=True)

model.predict(source="C:\\Users\\harsh\\Downloads\\WhatsApp Video 2024-03-21 at 2.05.33 PM.mp4" , show=True , save=True , conf=0.5)

cv2.waitKey(0)