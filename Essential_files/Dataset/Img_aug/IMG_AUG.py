import numpy as np
import cv2
import imgaug.augmenters as iaa
from utils import xywhn2xyxy, draw_img, seq_bbs_format, read_txt,read_img,xyxy2xywhn,save_img_aug,save_txt_aug
import os

image_dir='../datasets/Leaf&Tomatoes/train/images'
label_dir='../datasets/Leaf&Tomatoes/train/labels'
#image_dir='../small_test_file/images'
#label_dir='../small_test_file/labels'
aug_image_dir='../AUG_640/images'
aug_label_dir='../AUG_640/labels'
AUGLOOP=30
w1,h1=640,640
print(w1,h1)

###AUG methods you can add more processing methods listed by imgaug.augmenters
seq = iaa.Sequential([
    iaa.Fliplr(0.6),  #50% flip
    iaa.Sometimes(0.2,iaa.GaussianBlur(sigma=(0,0.2))),
    #iaa.Resize({'height':416, 'width':416}),
    iaa.LinearContrast((0.75, 1.25)),
    #iaa.Affine(
       # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
       # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
      #  rotate=(-5, 10),
      #  shear=(-4, 4))
    #iaa.Flipud(0.5)
    ])
#print(label_dir)

for root,sub_folders,files in os.walk(label_dir):
    print(label_dir)
    files[:] = [value for value in files if value != '.DS_Store']

    for name in files:
        label_path=os.path.join(label_dir,name)
        # label_path='../datasets/Leaf_Raw/train/labels/IMG_1184_JPG.rf.3bdfbb00b73e25e509325c0910979e0b.txt'
        # image_path='../datasets/Leaf_Raw/train/images/IMG_1184_JPG.rf.3bdfbb00b73e25e509325c0910979e0b.jpg'
        print(name)
        clss,bboxs_raw=read_txt(label_path)
        img_path=os.path.join(image_dir,str(name[:-4]+'.jpg'))
        img=read_img(img_path)
        bboxs_xyxy=xywhn2xyxy(bboxs_raw,w=w1,h=h1)
        bbs=seq_bbs_format(bboxs_xyxy,img,clss)        #
        for epoch in range(AUGLOOP):
            seq_det = seq.to_deterministic()
            aug_img = seq_det.augment_images([img])[0]   ###img aug

            new_bbx = []
            for i in range(len(bbs)):
                bbs_rescaled = seq_det.augment_bounding_boxes(bbs[i])  ####ANO aug
                # print(bbs[i])
                # print(bbs_rescaled)
                new_bbx.append((float(bbs_rescaled[0].x1), bbs_rescaled[0].y1, bbs_rescaled[0].x2,
                                bbs_rescaled[0].y2))
            print('nex_bbx:',new_bbx)
            aug_txt_xywh = xyxy2xywhn(new_bbx,w=w1,h=h1)
            print('aug_txt_xywh:', aug_txt_xywh)
            #aug_txt_xywh=xyxy2xywhn(new_bbx)         #
            aug_txt_value=np.hstack((clss,aug_txt_xywh))
            save_img_aug(aug_img,aug_image_dir,name,epoch)
            save_txt_aug(aug_txt_value,aug_label_dir,name,epoch)

