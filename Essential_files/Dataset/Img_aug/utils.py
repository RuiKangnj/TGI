#yolo convert to x1,y1,x2,y2
import torch
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
ia.seed(1)
def xywhn2xyxy(x, w, h, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

###x1,y1,x2,y2转为yolo
def xyxy2xywhn(x, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    x=np.asarray(x)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #print(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color

def seq_aug():
    seq = iaa.Sequential([
        iaa.Fliplr(1),  #50% flip
        iaa.Resize({'width':640, 'height':640})
    ])
    return (seq)

def seq_bbs_format(bbox,img,clss):
    new_bbs=[]
    for i in range(len(bbox)):
        bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=int(bbox[i][0]), y1=int(bbox[i][1]), x2=int(bbox[i][2]), y2=int(bbox[i][3]),label=int(clss[i]))],shape=img.shape)
        new_bbs.append(bbs)
    return(new_bbs)

def read_img(path):
    img=cv2.imread(path)
    return (img)
def draw_img(img,bbox,clss):

    for i in range(len(clss)):
        class_id=int(clss[i])
        color=get_id_color(class_id)
        x1, y1, x2, y2=int(bbox[i][0]),int(bbox[i][1]),int(bbox[i][2]),int(bbox[i][3])

        drawed_image = cv2.rectangle( img, (x1, y1),(x2, y2), color, thickness=2 )
    return(drawed_image)

def read_txt(label_path):
    ls=[]
    with open(label_path, 'r') as ano_list:  #
        for lst in ano_list.readlines():
            lst = lst.encode('utf-8').strip()  #
            lst = lst.split()  #
            p = [float(i) for i in lst]
            ls.append(p)
            # ob_class=lst.strip()
        ls = np.array(ls)

    #from utils import xywhn2xyxy, draw_img, seq_bbs_format
    bboxs = ls[:, [1, 2, 3, 4]]
    clss = ls[:, [0]]
   # bboxs_xyxy = xywhn2xyxy(bboxs, w, h)
    return(clss,bboxs)

def save_img_aug(aug_img,aug_img_dir,name,epoch):
    aug_img_path=os.path.join(aug_img_dir,str(name[:-4]+'_aug_'+str(epoch)+'.jpg'))
    #aug_img_path = os.path.join(aug_img_dir, str(name[:-4]  + '.jpg'))
    cv2.imwrite(aug_img_path,aug_img)

def save_txt_aug(aug_txt_value,aug_label_dir,name,epoch):
    aug_txt_path = os.path.join(aug_label_dir, str(name[:-4] + '_aug_' + str(epoch) + '.txt'))
    np.savetxt(aug_txt_path,aug_txt_value,delimiter=' ')