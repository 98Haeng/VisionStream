from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from checker import ObjectDetector
import pickle
import warnings
warnings.filterwarnings('ignore')
import requests

app = FastAPI()

# 요청 스키마 정의
class PredictionRequest(BaseModel):
    video: str  # Base64 인코딩된 비디오 데이터

# 객체 검출기 인스턴스 생성
detector = ObjectDetector(num_classes=130, model_type='yolo_nas_l', weight_path='ckpt_best.pth', confidence_threshold=0.5)

# 비디오를 Base64에서 NumPy 배열로 변환하는 함수
def video_to_np_array(video_data):
    # Base64 디코딩
    video_bytes = base64.b64decode(video_data)
    # 비디오 데이터를 NumPy 배열로 변환
    video_np = np.frombuffer(video_bytes, dtype=np.uint8)
    return video_np

# 결과 리턴 양식 지정
def format_prediction(detected_objects, img_id='uploaded_video'):
    outputs = {'items': []}
    for obj in detected_objects:
        outputs['items'].append({
            'label': obj['label'],
            'conf': float(obj['confidence']),
            'xcenter': obj['x_center'],
            'ycenter': obj['y_center'],
            'xstart': int(obj['box'][0]),
            'ystart': int(obj['box'][1]),
            'xend': int(obj['box'][2]),
            'yend': int(obj['box'][3]),
            'korean': obj['korean'],
        })
    return outputs

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # 비디오 데이터를 NumPy 배열로 변환
        video_np = video_to_np_array(request.video)
        # 비디오 프레임 추출
        cap = cv2.VideoCapture(video_np.tobytes())
        detected_objects = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 객체 검출 실행
            detected_objects.extend(detector.detect_objects(frame))
        # 결과 포매팅
        formatted_output = format_prediction(detected_objects)
        return JSONResponse(content=formatted_output)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 서버 실행에 필요한 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)