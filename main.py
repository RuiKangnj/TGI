import copy
import time
import argparse

import cv2,math
from pathlib import Path

from Detector.detector import ObjectDetector
from Tracker.tracker import MultiObjectTracker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default='videos/tomato2.mp4')

    parser.add_argument(
        '--detector',
        choices=[
            'yolox_n',
            'yolox_s',
            'yolox_m',
            'yolo_tgi_n',
            'yolo_tgi_s',
            'yolo_tgi_m',
            'nanodet_n',
            'nanodet_s',
            'nanodet_m',

        ],
        default='yolo_tgi_s',
    )
    parser.add_argument(
        '--tracker',
        choices=[
            'motpy',
            'mc_bytetrack',
            'mc_norfair',
        ],
        default='mc_bytetrack',
    )

    parser.add_argument("--target_id", type=str, default='2')

    parser.add_argument('--use_gpu', action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie

    detector_name = args.detector
    tracker_name = args.tracker

    target_id = args.target_id
    if target_id is not None:
        target_id = [int(i) for i in target_id.split(',')]

    use_gpu = args.use_gpu

    # Video ini
    cap = cv2.VideoCapture(cap_device)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)              ###fps
    # Object Detection   loading Detect model
    detector = ObjectDetector(
        detector_name,
        target_id,
        use_gpu=use_gpu,
    )
 #   detector.print_info()

    # Multi Object Tracking     loading tracking model
    tracker = MultiObjectTracker(
        tracker_name,
        cap_fps,
        use_gpu=use_gpu,
    )
 #   tracker.print_info()
    track_id_dict = {}
    filtered_ids_dict={}
    id_no = 0
    center_tracks={}
    save_path = 'demo'
    N, img = cap.read()
    fps, w, h = cap_fps, img.shape[1], img.shape[0]
    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        #frame=cv2.flip(cv2.transpose(frame), 0)
        debug_image = copy.deepcopy(frame)

        frame_h,frame_w,_=debug_image.shape
        x_min=int(frame_w/2-250)
        x_max=int(frame_w/2+250)
        #print('h',frame_h,'w',frame_w)

        d_bboxes, d_scores, d_class_ids = detector(frame)         ###获得（bbox,scores,class_id）
        # Multi Object Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        #print('track_ids',track_ids,len(track_ids))
        t,filtered_id_dict= total_count(track_ids, t_bboxes, x_min,x_max,id_no,filtered_ids_dict)
        #print('filtered_id_dict',filtered_id_dict)
        total_numbers = len(filtered_id_dict)


        # link track_id
        for track_id in track_ids:
            if track_id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[track_id] = new_id

        #print('track_id_dict)',track_id_dict,len(track_id_dict))

        elapsed_time = time.time() - start_time

        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            track_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            track_id_dict,
            #filtered_id_dict,
            total_numbers,
            x_min,
            x_max,
            frame_h,
            center_tracks
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('Detection and tracking', debug_image)
        vid_writer.write(debug_image)
    vid_writer.release()
   # cap.release()
   # cv2.destroyAllWindows()

def get_id_color(index):
    #temp_index = abs(int(index + 1)) * 3
    if index == 1:
        color=(0,255,0)
    elif index == 2:
        color=(255,0,0)
    elif index ==3 :
        color=(255,255,0)
    text_color_bgr = tuple(reversed(color))
    return text_color_bgr

def total_count(track_ids, bboxes, x_min,x_max,id_no,filtered_ids_dict):
    for i in range(len(bboxes)):
        x1, y1, x2, y2= bboxes[i]
    # 撞线检测点，(x1，y1)，x方向偏移比例 0.0~1.0
        x_center= int(x1 + ((x2 - x1) * 0.5))
      # 撞线的点
        x = x_center
        if x > x_min and x < x_max:
            if track_ids[i] not in filtered_ids_dict:
                filtered_ids_dict[track_ids[i]]=id_no
                id_no += 1
           # elif track_ids[i] in filtered_ids_dict:
            #    filtered_ids_dict.remove(track_ids[i])

    return (id_no,filtered_ids_dict)
    #return(filtered_ids_dict,bboxes_dict,scores_dict,class_ids_dict,count_no)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)



def draw_debug_info(
    debug_image,
    elapsed_time,
    track_ids,
    bboxes,
    scores,
    class_ids,
    track_id_dict,
    count_no,
    x_min,
    x_max,
    frame_h,
    center_tracks,

):
    #for id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
        ####filter here
    data= zip(track_ids, bboxes, scores, class_ids)
    print('class_ids:',class_ids)
   # data_sorted=sorted(data, key=lambda x: x[1][0],reverse=True)
    #print(bboxes)
    for id, bbox, score, class_id in data:

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_id=int(class_id)
        #color = get_id_color(track_id_dict[id])
        print('class_id:'+str(class_id))
        color = get_id_color(class_id)
       # color = get_id_color(2)
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=3,
        )

        center = (int((x1 + x2) / 2), int((y1+y2) / 2))  # BBox中心点
        debug_image=cv2.circle(debug_image,center, 2, (0, 255, 0), -1)
        if id not in center_tracks:
            center_tracks[id]=[]
        center_tracks[id].append(center)
        # 定义距离阈值绘制轨迹
        distance_threshold = 50
        #for i in range(1, len(center_tracks[id])):
           #if distance(center_tracks[id][i - 1], center_tracks[id][i]) < distance_threshold:
            # 绘制轨迹线
              #  cv2.line(debug_image, center_tracks[id][i - 1], center_tracks[id][i], (0, 0, 255), 3)
        #score = '%.2f' % score
        #text = 'TID:%s(%s)' % (str(int(track_id_dict[id])), str(score))
        #text = 'TID:%s(%s)' % (str(int(2)), str(score))
        #debug_image = cv2.putText(debug_image,text,(x1, y1 - 22),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness=3)
        ###video save


       # text = 'CID:%s' % (str(int(class_id)))
        #debug_image = cv2.putText( debug_image,text,(x1, y1 - 8),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,thickness=3)

    # inference time
    cv2.putText(debug_image,"Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        debug_image,
        "Total numbers : " + '{:.1f}'.format(count_no) ,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.line(debug_image, (x_min, 0), (x_min, frame_h), (0, 255, 255), 10)
    cv2.line(debug_image, (x_max, 0), (x_max, frame_h), (0, 255, 255), 10)

    return debug_image


if __name__ == '__main__':
    main()
