import argparse
import sys
stdout = sys.stdout
# import super_gradients
import cv2
import numpy as np
import random
import torch
from super_gradients.training import models
import pickle
import json

sys.stdout = stdout

class ObjectDetector:
    def __init__(self, num_classes, model_type, weight_path, confidence_threshold=0.5):
        self.num_classes = num_classes
        self.model_type = model_type
        self.weight_path = weight_path
        self.confidence_threshold = confidence_threshold

        # 모델 로드
        self.model = models.get(self.model_type, num_classes=self.num_classes, checkpoint_path=self.weight_path)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def detect_objects(self, img_np):
        # 바이트 데이터를 NumPy 배열로 변환
        # NumPy 배열을 이미지로 변환
        with open('km.json', 'r', encoding='utf-8') as file:
            key_mappings = json.load(file)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Checker에서 이미지 디코딩이 불가능합니다.")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 객체 검출 수행
        preds = self.model.predict(img_rgb, conf=self.confidence_threshold)._images_prediction_lst[0]
        dp = preds.prediction
        bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

        class_names = preds.class_names
        detected_objects_info = []
        image_name = 'detected_image'
        for box, conf, label in zip(bboxes, confs, labels):
            label_name = class_names[int(label)]
            korean_name = key_mappings.get(label_name, '알 수 없음')
            detected_object = {
                'image_name': image_name,
                'label': label_name,
                'confidence': conf,
                'x_center': int((box[0] + box[2]) / 2),
                'y_center': int((box[1] + box[3]) / 2),
                'box': box,
                'korean': korean_name
            }
            detected_objects_info.append(detected_object)
            self._plot_one_box(box, img, label=f"{label_name} {conf:.3f}")

        # 모든 객체 정보를 파일에 기록
        with open('detected_objects.txt', 'w') as f:
            for obj in detected_objects_info:
                f.write(f"{obj['label']}, {obj['x_center']}, {obj['y_center']}, {obj['confidence']:.2f}, {obj['korean']}\n")

        # 바운딩 박스가 표시된 이미지 저장
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, required=True, help="number of classes the model trained on")
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s', choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'], help="Model type (eg: yolo_nas_s)")
    ap.add_argument("-w", "--weight", type=str, required=True, help="path to trained model weight")
    ap.add_argument("-i", "--image", type=str, required=True, help="image path")
    ap.add_argument("-c", "--conf", type=float, default=0.25, help="model prediction confidence (0<conf<1)")
    args = vars(ap.parse_args())

    # 이미지 파일을 읽어서 pickle로 직렬화
    with open(args['image'], "rb") as image_file:
        # 이미지 데이터를 메모리에 로드
        image_data = image_file.read()
        # print(image_data)
        # 이미지 데이터를 pickle 형식으로 직렬화
        pickled_image_data = pickle.dumps(image_data)
        # print(pickled_image_data)

    # pickle 데이터를 역직렬화하여 원본 이미지 데이터를 복원
    image_restored = pickle.loads(pickled_image_data)
    # print(image_restored)

    # NumPy 배열로 변환
    image_np = np.frombuffer(image_restored, dtype=np.uint8)
    # print(image_np)

    # 객체 검출기 인스턴스 생성 및 객체 검출 실행
    detector = ObjectDetector(num_classes=args['num'], model_type=args['model'], weight_path=args['weight'], confidence_threshold=args['conf'])
    detected_objects = detector.detect_objects(image_np)