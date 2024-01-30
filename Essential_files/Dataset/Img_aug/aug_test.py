import numpy
import cv2
import imgaug.augmenters as iaa
import imgaug as ia
###This script is used to validate the annotations correctly aligned with images after augmentation

image_path='../datasets/Leaf_Raw/train/images/IMG_1184_JPG.rf.3bdfbb00b73e25e509325c0910979e0b.jpg'
label_path='../datasets/Leaf_Raw/train/labels/IMG_1184_JPG.rf.3bdfbb00b73e25e509325c0910979e0b.txt'
ls=[]
img=cv2.imread(image_path)
h,w=img.shape[0],img.shape[1]
print(h,w)
with open(label_path, 'r') as ano_list: # train.tx or test.txt，including img paths
    for lst in ano_list.readlines():
        lst=lst.strip()     #
        lst=lst.split()     #
        p=[float(i) for i in lst]
        ls.append(p)
        #ob_class=lst.strip()
    ls=numpy.array(ls)

from utils import  xywhn2xyxy,draw_img,seq_bbs_format,xyxy2xywhn

bboxs=ls[:,[1,2,3,4]]
print(bboxs)
clss=ls[:,[0]]
bboxs_xyxy=xywhn2xyxy(bboxs,w,h)
print('raw',bboxs_xyxy)
###creat seq
seq = iaa.Sequential([
       # iaa.Fliplr(1),  #50% flip
        iaa.Resize({'width':640, 'height':640})
    ])
###augment images and bbs
bbs=seq_bbs_format(bboxs_xyxy,img,clss)
#print(len(bbs))
seq_det = seq.to_deterministic()
img_augment=seq_det.augment_images([img])[0]
new_bbx=[]
for i in range(len(bbs)):
    bbs_rescaled=seq_det.augment_bounding_boxes(bbs[i])     ####ANO aug
   # print(bbs[i])
   # print(bbs_rescaled)
    new_bbx.append([bbs_rescaled[0].x1, bbs_rescaled[0].y1, bbs_rescaled[0].x2,
                    bbs_rescaled[0].y2])
    #new_bbx.append([bbs_rescaled[0].x1,bbs_rescaled.bounding_boxes[0].y1,bbs_rescaled.bounding_boxes[0].x2,bbs_rescaled.bounding_boxes[0].y2])

print('new_bbx',new_bbx)
#print(len(bboxs_xyxy))
#print(len(new_bbx))
###测试xyxy2xywh,xywh2xyxy
bbox_xywh=xyxy2xywhn(new_bbx,w=640,h=640)
print('xywh',bbox_xywh)
aug_bbox_xyxy=xywhn2xyxy(bbox_xywh,w=640,h=640)
new_bbx=aug_bbox_xyxy
#
#
seq_det = seq.to_deterministic()
img_augment=seq_det.augment_images([img])[0]

cv2.imshow('img_aug',img_augment)

cv2.imshow('raw_img',img)

raw_drawed_img=draw_img(img,bboxs_xyxy,clss)

cv2.imshow('raw_drawed',raw_drawed_img)

aug_drawed_img=draw_img(img_augment,new_bbx,clss)
cv2.imshow('aug_drawed',aug_drawed_img)
k=cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()

