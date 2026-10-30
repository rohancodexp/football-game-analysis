import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser()
p.add_argument('--data',required=True)
p.add_argument('--epochs',type=int,default=50)
p.add_argument('--device',default='cpu')
a=p.parse_args()
YOLO('yolov8n.pt').train(data=a.data,epochs=a.epochs,imgsz=960,batch=4,device=a.device,project='runs',name='player_detector',exist_ok=False)
