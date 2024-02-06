# TGI
Detector and tracker for leaf disease detection and tomato counting.
### Single image inference
You could check draw_inferences.ipynb to see how to inference a single img or img folder.
### video stream inference
run python main.py --movie {your tomato video path} or change the settings in "args"
#### demo video
(https://drive.google.com/file/d/1OtbYlxib0nB0MXR31Ha35pH_M6MAC9YM/view?usp=sharing)
### the onnx weights of base models
Can be reached in the link below:
(https://drive.google.com/file/d/1G1E-6ADPoj9g6SUZs6URc2-tWRqulQ5I/view?usp=sharing)

Download and unzip onnx files, and move them to corresponding detector folders, the path of Nanodet for example: Detector/nanodet/model/Nanodet_M.onnx .

# Dataset
The High-quality annotated images of tomato leaves are available for download from the Roboflow link https://universe.roboflow.com/jaas/leaf-and-tomato. To facilitate the expansion of TGI networks for detection research on custom datasets, we provide image augmentation codes.

The augmentation codes can be found in the folder Essential_files/Dataset/Img_aug, where the IMG_AUG.py file is used to generate augmented images, and other codes are for post-aug images validation and dataset preparation. The utils.py file contains definitions for various processing operations, while the Format/ includes codes for yolo2voc, xml2coco, and yolo2xml conversions.

We provided a tomato & leaf diseases aug dataset for model training, the labels were summarized in different formats, which can be downloaded from (https://drive.google.com/file/d/1U--z8pbU7yUIXOjxxYO5MHAFXaQSqt5V/view?usp=drive_link).

we also provided several tomato scanning videos to test the tracking algorithms, the counting numbers were summarized in the (.csv) file, these videos can be downloaded from (https://drive.google.com/drive/folders/1JtOEmRp1vKw0-I6JN3RGuMhxzEedkGq0?usp=drive_link).

# TGI-Detector training
For the TGI-Detector model, it is based on modifications to the YOLOv8 network(https://github.com/ultralytics/ultralytics). Researchers can download our tgi_modules files to replace the original files in the official ultralytics/nn/ folder.

For detector network training and model export, refer to the yolo-tgi-trail.py script. The leaf_tomato_aug.yaml includes image path definitions required for network training, and yolo-tgi.yaml contains parameters for the network's n, s, m scales.

We also provide a tgi.pt file for the m scale, which others can use to fine-tune their custom datasets. 
