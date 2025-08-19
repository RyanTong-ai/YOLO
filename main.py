from ultralytics import YOLO
import time
import cvzone
import math
import numpy as np
import cv2
Prv_time=0
capture= cv2.VideoCapture(0)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
model=YOLO("../Yolo-Weights/yolov8n.pt")  # Path to your YOLOv8 model
while True:
    ret,frame= capture.read()
    width=int(capture.get(3))
    height=int(capture.get(4))
    current_time=time.time()
    fps=1/(current_time-Prv_time)
    Prv_time=current_time
    cv2.putText(frame,f'FPS:{fps:.2g}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
     
    results=model(frame, stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int (x1),int(y1),int(x2),int(y2) # Convert to int
            print(f"Coordinates: {x1}, {y1}, {x2}, {y2}")
            cv2.rectangle(frame, (x1,y1) ,(x2,y2) ,(0, 255, 0), 2)  # Draw bounding box
            
            conf=math.ceil(box.conf[0]*100)/100
            print(conf)
            
            cls=int(box.cls[0])
            cvzone.putTextRect(frame,f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1+35)), scale=1, thickness=2, offset=3, colorR=(0, 255, 0))
            
        # Use streaming for better performance
    '''image=np.zeros(frame.shape,dtype=np.uint8)
    
    
    smaller_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    smaller_frame=cv2.rectangle(smaller_frame,(0,0),(width,height),(255,0,0),5)
    
    image[0:height//2,0:width//2]=smaller_frameq
    image[height//2:height,width//2:width]=smaller_frame
    image[0:height//2,width//2:width]=smaller_frame
    image[height//2:height,0:width//2]=smaller_frame
    
    font=cv2.FONT_HERSHEY_SIMPLEX
    img=cv2.putText(image,'Frame',(200 ,height-20),font,4,(0,0,0),5,cv2.LINE_AA)''' #code to create a grid of small frames
    cv2.imshow('Webcam',frame)
    
    if cv2.waitKey(100) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
     

