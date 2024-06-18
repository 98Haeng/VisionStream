import argparse
import os
import cv2
import torch
import numpy as np
import random
import time
from super_gradients.training import models
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num", type=int, required=True, help="모델이 훈련된 클래스 수")
ap.add_argument("-m", "--model", type=str, default='yolo_nas_s', choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], help="모델 타입 (예: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, required=True, help="훈련된 모델 가중치 경로")
ap.add_argument("-s", "--source", type=str, required=True, help="비디오 경로/카메라-id/RTSP")
ap.add_argument("-c", "--conf", type=float, default=0.25, help="모델 예측 신뢰도 (0<conf<1)")
ap.add_argument("--save", action='store_true', help="비디오 저장")
ap.add_argument("--hide", action='store_false', help="추론 창 숨기기")
args = vars(ap.parse_args())

# 이전 데이터가 있는지 확인하고 초기화
stats_file_path = os.path.join('runs', 'detect', 'stats.csv')
if os.path.exists(stats_file_path):
    os.remove(stats_file_path)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_bbox(img, frame_count, stats_data):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = model.predict(img_rgb, conf=args['conf'])._images_prediction_lst[0]
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    
    interested_labels = ['sleep', 'activity', 'seeteacher', 'blankout', 'somethingelse', 'snooze', 'seeother', 'seebook', 'seemonitor']
    label_counts = {}
    label_confs = {}
    total_count = 0  # 전체 개수를 저장할 변수 초기화
    
    for box, cnf, label in zip(bboxes, confs, labels):
        class_name = class_names[int(label)]
        if class_name in interested_labels:
            plot_one_box(box[:4], img, label=f'{class_name} {cnf:.2f}', color=colors[label])
            if class_name in label_counts:
                label_counts[class_name] += 1
                label_confs[class_name].append(cnf)
            else:
                label_counts[class_name] = 1
                label_confs[class_name] = [cnf]
            total_count += 1  # 관심 라벨이면 전체 개수 증가

    if frame_count % 10 == 0 and label_counts:
        for label in label_counts:
            avg_conf = sum(label_confs[label]) / label_counts[label]
            stats_data.append([frame_count, label, label_counts[label], avg_conf, total_count])

# YOLO-NAS 모델 로드
model = models.get(args['model'], num_classes=args['num'], checkpoint_path=args["weight"])
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=args['conf'])._images_prediction_lst[0].class_names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

global_timer = time.time()

stats_data = []
csv_columns = ['Frame', 'Label', 'Count', 'Average Confidence', 'Total Count']

if args['source'].endswith('.jpg') or args['source'].endswith('.jpeg') or args['source'].endswith('.png'):
    img = cv2.imread(args['source'])
    get_bbox(img, 0, stats_data)
    
    if stats_data:
        pd.DataFrame(stats_data, columns=csv_columns).to_csv(stats_file_path, index=False)
    
    print(f'[INFO] {(time.time()-global_timer)/60:.2f}분 동안 완료되었습니다.')
    
    if not args['hide']:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    if args['save']:
        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        path_save = os.path.join('runs', 'detect', os.path.split(args['source'])[1])
        cv2.imwrite(path_save, img)
        print(f"[INFO] 이미지 저장: {path_save}")

else:
    video_path = args['source']
    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if args['save'] or not args['hide']:
        original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        if not str(video_path).isnumeric():
            path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
        else:
            c = 0
            while True:
                if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
                    path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
                    break
                c += 1
        out_vid = cv2.VideoWriter(path_save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (original_video_width, original_video_height))

    p_time = 0
    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] 읽기 실패...')
            break
        
        get_bbox(img, frame_count, stats_data)
        
        if frame_count % 10 == 0 and stats_data:
            pd.DataFrame(stats_data, columns=csv_columns).to_csv(stats_file_path, index=False, mode='a', header=not os.path.exists(stats_file_path))
            stats_data.clear()

        frame_count += 1

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {fps:.3f}', (50, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if args['save'] or not args['hide']:
            out_vid.write(img)

        if args['hide']:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print(f'[INFO] {(time.time()-global_timer)/3600:.2f}시간 동안 완료되었습니다.')
    cap.release()
    if args['save'] or not args['hide']:
        out_vid.release()
        print(f"[INFO] 출력 비디오 저장: {path_save}")
    if args['hide']:
        cv2.destroyAllWindows()

