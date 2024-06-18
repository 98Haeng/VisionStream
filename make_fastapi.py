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
    image: str  # Base64 인코딩된 이미지 데이터
# 객체 검출기 인스턴스 생성
detector = ObjectDetector(num_classes=130, model_type='yolo_nas_l', weight_path='ckpt_best.pth', confidence_threshold=0.5)

# 이미지를 Base64에서 NumPy 배열로 변환하는 함수
def image_to_np_array(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    # 이미지 데이터를 pickle 형식으로 직렬화
    pickled_image_data = pickle.dumps(image_data)
    # pickle 데이터를 역직렬화하여 원본 이미지 데이터를 복원
    image_restored = pickle.loads(pickled_image_data)
    # NumPy 배열로 변환
    image_np = np.frombuffer(image_restored, dtype=np.uint8)
    return image_np

# 결과 리턴 양식 지정
def format_prediction(detected_objects, img_id='uploaded_image'):
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
        # 이미지 데이터를 NumPy 배열로 변환
        img_np = image_to_np_array(request.image)
        # 객체 검출 실행
        detected_objects = detector.detect_objects(img_np)
        # 결과 포매팅
        formatted_output = format_prediction(detected_objects)
        return JSONResponse(content=formatted_output)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 서버 실행에 필요한 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




#### 비디오, 이미지 구분
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import io
# from PIL import Image
# import cv2
# import torch
# from torchvision import transforms

# app = FastAPI()

# # 모델 로드 함수
# def load_model(checkpoint_path):
#     # 모델 로딩은 여러분의 모델 아키텍처에 맞게 조정해야 합니다.
#     model = torch.load(checkpoint_path)
#     model.eval()
#     return model

# model = load_model("path_to_your_model.ckpt")

# # 이미지 전처리 함수
# def transform_image(image):
#     my_transforms = transforms.Compose([
#         transforms.Resize(255),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()
#     ])
#     image = Image.open(io.BytesIO(image))
#     return my_transforms(image).unsqueeze(0)

# # 동영상을 프레임으로 변환
# def video_to_frames(video_bytes):
#     video_buffer = io.BytesIO(video_bytes)
#     video_buffer.seek(0)
#     cap = cv2.VideoCapture(video_buffer.read())
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(frame)
#         tensor = transform_image(pil_image.tobytes())
#         frames.append(tensor)
#     cap.release()
#     return frames

# # 예측 및 결과 포매팅 함수
# def format_prediction(detected_objects, img_id):
#     outputs = {'items': []}
#     for obj in detected_objects:
#         outputs['items'].append({
#             'image_name': img_id,
#             'label': obj['label'],
#             'conf': obj['confidence'],
#             'xcenter': obj['x_center'],
#             'ycenter': obj['y_center'],
#             'xstart': int(obj['box'][0]),
#             'ystart': int(obj['box'][1]),
#             'xend': int(obj['box'][2]),
#             'yend': int(obj['box'][3]),
#             'korean': obj['korean'],
#         })
#     return outputs

# # 엔드포인트 정의
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     file_bytes = await file.read()
#     if file.content_type.startswith('image/'):
#         tensor = transform_image(file_bytes)
#         tensors = [tensor]
#     elif file.content_type.startswith('video/'):
#         tensors = video_to_frames(file_bytes)
#     else:
#         return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

#     results = []
#     for tensor in tensors:
#         with torch.no_grad():
#             detected_objects = model(tensor)
#         formatted_output = format_prediction(detected_objects, file.filename)
#         results.append(formatted_output)

#     return JSONResponse(content={"results": results})

# # 서버 실행 코드
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)