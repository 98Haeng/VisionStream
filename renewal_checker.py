# import argparse
# import sys
# stdout = sys.stdout
# # import super_gradients
# import cv2
# import numpy as np
# import random
# import torch
# from super_gradients.training import models
# import pickle
# import json

# sys.stdout = stdout

# class ObjectDetector:
#     def __init__(self, num_classes, model_type, weight_path, confidence_threshold=0.5):
#         self.num_classes = num_classes
#         self.model_type = model_type
#         self.weight_path = weight_path
#         self.confidence_threshold = confidence_threshold

#         # 모델 로드
#         self.model = models.get(self.model_type, num_classes=self.num_classes, checkpoint_path=self.weight_path)
#         self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

#         with open('key_mappings.json', 'r', encoding='utf-8') as file:
#             self.key_mappings = json.load(file)

#     def detect_objects(self, video_np):
#         cap = cv2.VideoCapture(video_np.tobytes())
#         detected_objects = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # 객체 검출 실행
#             detected_objects.extend(self._detect_objects_in_frame(frame))
#         return detected_objects

#     def _detect_objects_in_frame(self, frame):
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         preds = self.model.predict(img_rgb, conf=self.confidence_threshold)._images_prediction_lst[0]
#         dp = preds.prediction
#         bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
#         class_names = preds.class_names
#         best_objects = {}
#         image_name = 'detected_image'

#         for box, conf, label in zip(bboxes, confs, labels):
#             label_name = class_names[int(label)]
#             korean_name = self.key_mappings.get(label_name, '알 수 없음')
#             if label_name not in best_objects or best_objects[label_name]['confidence'] < conf:
#                 best_objects[label_name] = {
#                     'image_name': image_name,
#                     'label': label_name,
#                     'confidence': conf,
#                     'x_center': int((box[0] + box[2]) / 2),
#                     'y_center': int((box[1] + box[3]) / 2),
#                     'box': box,
#                     'korean': korean_name
#                 }

#         detected_objects_info = list(best_objects.values())

#         return detected_objects_info

#     def _plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
#         tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#         color = color or [random.randint(0, 255) for _ in range(3)]
#         c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#         cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#         if label:
#             tf = max(tl - 1, 1)
#             t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#             c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#             cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
#             cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-n", "--num", type=int, required=True, help="number of classes the model trained on")
#     ap.add_argument("-m", "--model", type=str, default='yolo_nas_s', choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], help="Model type (eg: yolo_nas_s)")
#     ap.add_argument("-w", "--weight", type=str, required=True, help="path to trained model weight")
#     ap.add_argument("-v", "--video", type=str, required=True, help="video path")
#     ap.add_argument("-c", "--conf", type=float, default=0.25, help="model prediction confidence (0<conf<1)")
#     args = vars(ap.parse_args())

#    # 비디오 파일을 읽어서 pickle로 직렬화
#     with open(args['video'], "rb") as video_file:
#         video_data = video_file.read()
#         pickled_video_data = pickle.dumps(video_data)
#         video_restored = pickle.loads(pickled_video_data)
#         video_np = np.frombuffer(video_restored, dtype=np.uint8)

#     # 객체 검출기 인스턴스 생성 및 객체 검출 실행
#     detector = ObjectDetector(num_classes=args['num'], model_type=args['model'], weight_path=args['weight'], confidence_threshold=args['conf'])
#     detected_objects = detector.detect_objects(video_np)

#     # 결과 저장
#     with open('detected_objects.txt', 'w') as f:
#         for obj in detected_objects:
#             f.write(f"{obj['label']}, {obj['x_center']}, {obj['y_center']}, {obj['confidence']:.2f}, {obj['korean']}\n")
#     # 경고음 설정
#     concentrate_label = 'concentrate'
#     concentrate_confidence_threshold = 0.2
#     for obj in detected_objects:
#         if obj['label'] == concentrate_label and obj['confidence'] <= concentrate_confidence_threshold:
#             print("Warning: Concentrate label detected with confidence", obj['confidence'])

import argparse
import sys
import os
import time
import cv2
import numpy as np
import random
import pickle
import torch
from super_gradients.training import models

class ObjectDetector:
    def __init__(self, num_classes, model_type, weight_path, confidence_threshold=0.5):
        self.num_classes = num_classes
        self.model_type = model_type
        self.weight_path = weight_path
        self.confidence_threshold = confidence_threshold
        self.model = models.get(self.model_type, num_classes=self.num_classes, checkpoint_path=self.weight_path)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.key_mappings = {
            "sleep" : "잠듦",
            "blankout" : "멍때리기",
            "somethingelse" : "딴짓",
            "seebook" : "책보기",
            "watchteacher" : "선생님보기",
            "activity" : "수업참여",
            "seeother" : "딴데보기"
        }

    def detect_objects(self, img_np):
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = self.model.predict(img_rgb, conf=self.confidence_threshold)._images_prediction_lst[0]
        dp = preds.prediction
        bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

        class_names = preds.class_names
        best_objects = {}
        image_name = 'detected_image'

        for box, conf, label in zip(bboxes, confs, labels):
            label_name = class_names[int(label)]
            korean_name = self.key_mappings.get(label_name, '알 수 없음')
            if label_name not in best_objects or best_objects[label_name]['confidence'] < conf:
                best_objects[label_name] = {
                    'image_name': image_name,
                    'label': korean_name,
                    'confidence': conf,
                    'x_center': int((box[0] + box[2]) / 2),
                    'y_center': int((box[1] + box[3]) / 2),
                    'box': box
                }

        detected_objects_info = list(best_objects.values())

        for obj in detected_objects_info:
            self._plot_one_box(obj['box'], img, label=f"{obj['label']} {obj['confidence']:.3f}")

        cv2.imwrite('detected_image.jpg', img)
        return detected_objects_info

    def _plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, required=True, help="학습된 모델의 클래스 수")
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s', choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], help="모델 타입")
    ap.add_argument("-w", "--weight", type=str, required=True, help="학습된 모델 가중치 경로")
    ap.add_argument("-i", "--image", type=str, help="이미지 경로")
    ap.add_argument("-v", "--video", type=str, help="비디오 경로")
    ap.add_argument("-c", "--conf", type=float, default=0.25, help="모델 예측 신뢰도")
    ap.add_argument("-s", "--save", action='store_true', help="출력 비디오 저장")
    ap.add_argument("-d", "--display", action='store_true', help="비디오 처리 중 비디오 디스플레이 표시")
    args = vars(ap.parse_args())

    if args['image']:
        with open(args['image'], "rb") as image_file:
            image_data = image_file.read()
            pickled_image_data = pickle.dumps(image_data)
            image_restored = pickle.loads(pickled_image_data)
            image_np = np.frombuffer(image_restored, dtype=np.uint8)
            detector = ObjectDetector(num_classes=args['num'], model_type=args['model'], weight_path=args['weight'], confidence_threshold=args['conf'])
            detected_objects = detector.detect_objects(image_np)

    elif args['video']:
        cap = cv2.VideoCapture(args['video'])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detector = ObjectDetector(num_classes=args['num'], model_type=args['model'], weight_path=args['weight'], confidence_threshold=args['conf'])
            detected_objects = detector.detect_objects(cv2.imencode('.jpg', frame)[1])
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()