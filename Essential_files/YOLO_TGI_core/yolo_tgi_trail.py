#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info,get_flops_with_torch_profiler
import torch

from thop import profile
from torchstat import stat

#stat(model, (3, 416, 416))  #
# Configure the tracking parameters and run the tracker
#model = YOLO('yolov8_faster.yaml')
model = YOLO('yolov8s_ghost.yaml').model
input = torch.randn(1, 3, 640, 640)
flops, params = profile(model, inputs=(input, ))
print("FLOPs: ", flops)
print("Parameters: ", params)
#print(stat(model, (3, 640, 640)))
#print(model_info(model))
#print(get_flops_with_torch_profiler(model)
###convert to ONNX format
#model = YOLO('../ONNX_files/raw_weights/Yolov8_ghost.pt')
#model = YOLO('runs/detect/yolov8_m/weights/best.pt')
#model.export(format="onnx",imgsz=[640,640], opset=12)

#model = YOLO('tomato.pt')
#results = model.track(source="tomato2.mp4", conf=0.7, iou=0.5, show=True)
# Train the model
#model.tune(data='leaf_tomato.yaml',epochs=30,iterations=300,optimizer='AdamW')
#model.train(data='leaf_tomato_aug.yaml',batch=64,resume=True,workers=32, epochs=500, optimizer='Adam')