import argparse

import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime
import copy




class YOLOv8(object):
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(
            self,
            model_path='yolov8.onnx',
            input_shape=(640, 640),
            class_score_th=0.3,
            nms_th=0.45,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            class_score_th: Confidence threshold for filtering detections.
            nms_th: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = model_path
        self.input_shape = input_shape
        self.class_score_th = class_score_th
        self.nms_th = nms_th
        self.providers=providers
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        # Load the class names from the leaf_tomato dataset
        self.classes = ['Healthy', 'Tomato', 'Unhealthy']
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        # Generate a color palette for the classes
       # self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
    def __call__(self, image):

        temp_image = copy.deepcopy(image)
        #image_height, image_width = image.shape[0], image.shape[1]   # 前処理
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        img_data= self.preprocess(temp_image)
        # 推論実施
        results = self.onnx_session.run( None, {self.input_name: img_data}, )
        # 後処理

        bboxes, scores, class_ids = self.postprocess(self.img, results)
        return bboxes, scores, class_ids

    def preprocess(self,image):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = image

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        #img=self.img
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image,output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:

        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.class_score_th:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.class_score_th, self.nms_th)

        # Iterate over the selected indices after non-maximum suppression
        bboxes=[]
        scoress=[]
        class_idss=[]
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            bboxes.append(box)
            scoress.append(score)
            class_idss.append(class_id)
            # Draw the detection on the input image
           # self.draw_detections(input_image, box, score, class_id)

        ##
        #print('old_bbx',bboxes)
        new_bboxes=[]
        for single_box in bboxes:
            x1, y1, w, h = single_box
            x1, y1, x2, y2=x1,y1,x1+w, y1+w
            new_bboxes.append([x1,y1,x2,y2])
        new_bboxes=np.array(new_bboxes)
        bboxes=new_bboxes
       # print('bboxes',bboxes.shape)

        if len(class_idss) > 0:
            class_idss = np.array(class_idss) + 1  # 1始まりのクラスIDに変更
      #  print('class_idss', class_idss)
        # スコア閾値での抽出
        target_index = np.where(
            np.array(scoress) > self.class_score_th, True, False)

        if len(target_index) > 0:
            bboxes = bboxes[target_index]
            scoress = np.array(scoress)[target_index]
            class_idss = class_idss[target_index]

        return bboxes, scoress, class_idss

