#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import onnxruntime
import cv2
import numpy as np
import onnxruntime as ort
import math


class NanoDetONNX(object):
    # NanoDet後処理用定義
    REG_MAX = 7
    PROJECT = np.arange(REG_MAX + 1)

    # 標準化用定義
    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    MEAN = MEAN.reshape(1, 1, 3)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    STD = STD.reshape(1, 1, 3)

    def __init__(
        self,
        model_path='Nanodet_N.onnx',
        input_shape=(320, 320),
        class_score_th=0.35,
        nms_th=0.6,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    ):
        # 入力サイズ
        self.strides = (8, 16, 32, 64)
        self.input_shape = (input_shape[1], input_shape[0])

        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = self.onnx_session.get_outputs()[0].name
        #print(output_detail)


        self.classes = ['Healthy', 'Tomato', 'Unhealthy']
        self.num_classes = len(self.classes)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.net = ort.InferenceSession(model_path, so, providers=providers)
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])
        self.reg_max = int((self.net.get_outputs()[0].shape[-1] - self.num_classes) / 4) - 1
        self.project = np.arange(self.reg_max + 1)
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self._make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)

        # ストライド毎のグリッド点を算出
        self.grid_points =[]

    def __call__(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        # 前処理：標準化、リシェイプ
        resize_image, new_height, new_width, top, left = self._resize_image(
            temp_image)
        x = self._pre_process(resize_image)

        # 推論実行
        results = self.onnx_session.run(None,{self.input_name: x})[0].squeeze(axis=0)
       # print('results:',results.shape)

        # 後処理：NMS、グリッド->座標変換
        bboxes, scores, class_ids = self.post_process(results)

        # 後処理：リサイズ前の座標に変換
        ratio_height = image_height / new_height
        ratio_width = image_width / new_width
        for i in range(bboxes.shape[0]):
            bboxes[i, 0] = max(int((bboxes[i, 0] - left) * ratio_width), 0)
            bboxes[i, 1] = max(int((bboxes[i, 1] - top) * ratio_height), 0)
            bboxes[i, 2] = min(
                int((bboxes[i, 2] - left) * ratio_width),
                image_width,
            )
            bboxes[i, 3] = min(
                int((bboxes[i, 3] - top) * ratio_height),
                image_height,
            )
        class_ids = class_ids
       # class_ids = class_ids +1  # 1始まりのクラスIDに変更

        return bboxes, scores, class_ids        ####bboxes[xmin,ymin,xmax,ymax]

    def _make_grid(self, featmap_size, stride):
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()
        return np.stack((xv, yv), axis=-1)

    def _resize_image(self, image, keep_ratio=True):
        top, left = 0, 0
        new_height, new_width = self.input_shape[0], self.input_shape[1]

        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                new_height = self.input_shape[0]
                new_width = int(self.input_shape[1] / hw_scale)

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                left = int((self.input_shape[1] - new_width) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    0,
                    0,
                    left,
                    self.input_shape[1] - new_width - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                new_height = int(self.input_shape[0] * hw_scale)
                new_width = self.input_shape[1]

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                top = int((self.input_shape[0] - new_height) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    top,
                    self.input_shape[0] - new_height - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            resize_image = cv2.resize(
                image,
                self.input_shape,
                interpolation=cv2.INTER_AREA,
            )

        return resize_image, new_height, new_width, top, left

    def _pre_process(self, image):
        # 標準化
        image = image.astype(np.float32)
        image = (image - self.MEAN) / self.STD

        # リシェイプ
        image = image.transpose(2, 0, 1).astype('float32')
        image = image.reshape(-1, 3, self.input_shape[0], self.input_shape[1])

        return image


    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    def post_process(self, preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for stride, anchors in zip(self.strides, self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[
                                                                                           ind:(ind + anchors.shape[0]),
                                                                                           self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred = self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride

            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]

            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)

        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence

        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.class_score_th,
                                   self.nms_th)
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices.flatten()]
            confidences = confidences[indices.flatten()]
            classIds = classIds[indices.flatten()]
            return mlvl_bboxes, confidences, classIds+1
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])