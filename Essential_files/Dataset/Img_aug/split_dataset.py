
import os
import random
from shutil import copyfile

# 1. Create img path files
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
random.seed(1)

#dataset_dir = '../AUG_640'
raw_img_dir='../AUG_640/images'
raw_txt_dir='../AUG_640/labels'
# 2.Dataset save file
split_dir = '../AUG_640_split'
train_dir = os.path.join(split_dir, "train")
val_dir = os.path.join(split_dir, "valid")
test_dir = os.path.join(split_dir, "test")
# 3.split into train,val,test
train_pct = 7
valid_pct = 2
test_pct = 1


for root, dirs, files in os.walk(raw_img_dir):
    #print(files)
    files[:] = [value for value in files if value != '.DS_Store']
    for i in range(len(files)):
        #print(len(files))#
        if (i % 10) <= train_pct:
            raw_img_name = os.path.join(raw_img_dir, files[i])
            raw_txt_name = os.path.join(raw_txt_dir, files[i][:-4] + '.txt')
            train_img_name=os.path.join(train_dir+'/images',files[i])
            train_txt_name=os.path.join(train_dir+'/labels',files[i][:-4]+'.txt')
            copyfile(raw_img_name,train_img_name)
            copyfile(raw_txt_name, train_txt_name)
           # print(train_img_name,train_txt_name)
        if (i % 10) > train_pct and (i % 10) <= (train_pct+valid_pct):
            raw_img_name = os.path.join(raw_img_dir, files[i])
            raw_txt_name = os.path.join(raw_txt_dir, files[i][:-4] + '.txt')
            val_img_name=os.path.join(val_dir+'/images',files[i])
            val_txt_name = os.path.join(val_dir + '/labels', files[i][:-4] + '.txt')
            copyfile(raw_img_name, val_img_name)
            copyfile(raw_txt_name, val_txt_name)
          #  print(val_img_name,val_txt_name)
        if i%10 == 9:
            raw_img_name = os.path.join(raw_img_dir, files[i])
            raw_txt_name = os.path.join(raw_txt_dir, files[i][:-4] + '.txt')
            test_img_name=os.path.join(test_dir+'/images',files[i])
            test_txt_name = os.path.join(test_dir + '/labels', files[i][:-4] + '.txt')
            copyfile(raw_img_name, test_img_name)
            copyfile(raw_txt_name, test_txt_name)
           # print(test_img_name,test_txt_name)





