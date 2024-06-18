import os
import json
import sys
stdout = sys.stdout
import logging
import time
from a2osdk.restserver import A2OBaseHandler, A2OOptionHandler, A2ORestServer
from a2osdk.a2oeurekaclient import runeurekaclient
import _version
import base64
import pickle

import argparse
import cv2
import numpy as np
import random
import torch
from super_gradients.training import models
from checker import ObjectDetector
sys.stdout = stdout

logger = logging.getLogger('a2o')
num_classes = 157  # 모델이 훈련된 클래스의 수
model_type = 'yolo_nas_l'  # 사용할 모델의 타입 (defalut)
weight_path = 'runs/train1/ckpt_best.pth'  # 훈련된 모델 가중치 파일의 경로
# image_path = 'Data5/test/img2.JPG'  # 감지를 수행할 이미지 파일의 경로
confidence_threshold = 0.51  # 모델 예측의 신뢰도 임계값 (default)

# 객체 검출기 인스턴스 생성
detector = ObjectDetector(num_classes=num_classes, model_type=model_type, weight_path=weight_path, confidence_threshold=confidence_threshold)

image = "newData/test/img3.jpg"
with open(image, 'rb') as f:
    img = f.read()
input = {
        'ocr_svc':'yolonas-service',
        'ocr-type':'WORD',
        'items': [ 

            (image, img)
        ]        
    }
output = {'items': []}

# 입력 데이터 직렬화
aPickeData = pickle.dumps(input)

obj = pickle.loads(aPickeData)
detector = ObjectDetector(num_classes=num_classes, model_type=model_type, weight_path=weight_path, confidence_threshold=confidence_threshold)
for item in obj['items']:
    try:
        img_id, img_data = item  # 이미지 ID와 바이너리 데이터 추출
        print(img_id)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        print(img_np)
        detected_objects = detector.detect_objects(img_np)
    
        for obj in detected_objects:
            output['items'].append({
                'image_name': img_id,
                'label': obj['label'],
                'conf': obj['confidence'],
                'xcenter': obj['x_center'],
                'ycenter': obj['y_center']
            })
    except Exception as e:
        logger.error(f'Error processing image {item.get("id", "unknown")}: {str(e)}', exc_info=True)

# self.logger.info(f'Prediction completed. Elapsed time: {time.time() - ticks} seconds')    
output_serialized = pickle.dumps(output)
print(output)