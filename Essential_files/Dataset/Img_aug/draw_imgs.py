###This script is used to draw anns on image from AUG datasets
import numpy as np
import cv2
import imgaug.augmenters as iaa
from utils import xywhn2xyxy, draw_img, seq_bbs_format, read_txt,read_img,xyxy2xywhn,save_img_aug,save_txt_aug
import os

#image_dir='../AUG_640/images'
#label_dir='../AUG_640/labels'
image_dir='../AUG_640_split/test/images'
label_dir='../AUG_640_split/test/labels'
drawed_dir='Dataset_preparation/drawed_imgs'
w1,h1=640,640
for root,sub_folders,files in os.walk(label_dir):
    #if files['.DS_Store'] is True:
    files[:] = [value for value in files if value != '.DS_Store']
    #print(len(files))
    for name in files:
        drawed_path=os.path.join(drawed_dir,str('drawed_'+name[:-4]+'.jpg'))
        label_path=os.path.join(label_dir,name)
        print(name)
        clss,bboxs_raw=read_txt(label_path)       #####w1,h1
        bboxs_xyxy = xywhn2xyxy(bboxs_raw, w=w1, h=h1)
        img_path=os.path.join(image_dir,str(name[:-4]+'.jpg'))
        img=read_img(img_path)
        drawed_img = draw_img(img, bboxs_xyxy, clss)        #####bboxs_raw(x1,y1,x2,y2)
        cv2.imwrite(drawed_path, drawed_img)