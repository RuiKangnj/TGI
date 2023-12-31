# -*- coding: utf-8 -*-
import numpy as np

from Tracker.norfair.tracker import Detection
from Tracker.norfair.tracker import Tracker as NorfairTracker


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


class MultiClassNorfair(object):

    def __init__(
        self,
        fps=30,
        max_distance_between_points=30,
    ):
        self.fps = fps
        self.max_distance_between_points = max_distance_between_points
        self.tracker_dict = {}

    def __call__(self, _, bboxes, scores, class_ids):
        for class_id in np.unique(class_ids):
            if not int(class_id) in self.tracker_dict:
                self.tracker_dict[int(class_id)] = NorfairTracker(
                    distance_function=euclidean_distance,
                    distance_threshold=self.max_distance_between_points,
                )

        t_ids = []
        t_bboxes = []
        t_scores = []
        t_class_ids = []
        for class_id in self.tracker_dict.keys():
            target_index = np.in1d(class_ids, np.array(int(class_id)))

            if len(target_index) == 0:
                continue

            target_bboxes = np.array(bboxes)[target_index]
            target_scores = np.array(scores)[target_index]

            detections = []
            for bbox, score in zip(target_bboxes, target_scores):
                points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                points_score = np.array([score, score])

                detection = Detection(points=points, scores=points_score)
                detections.append(detection)

            results = self.tracker_dict[class_id].update(detections=detections)
            for result in results:
                x1 = result.estimate[0][0]
                y1 = result.estimate[0][1]
                x2 = result.estimate[1][0]
                y2 = result.estimate[1][1]

                t_ids.append(str(int(class_id)) + '_' + str(result.id))
                t_bboxes.append([x1, y1, x2, y2])
                t_scores.append(score)
                t_class_ids.append(int(class_id))

        return t_ids, t_bboxes, t_scores, t_class_ids
