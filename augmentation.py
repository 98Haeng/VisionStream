import albumentations as A
import cv2
import os
import shutil

# 이미지가 있는 디렉토리 경로
image_directory = "class0518/processtrain/images"
# 라벨이 있는 디렉토리 경로
label_directory = "class0518/processtrain/labels"

# 증강 기법별 파이프라인 정의
transformations = {
    "rgb_shift": A.Compose([A.RGBShift(p=1.0)]),
    "hue_saturation": A.Compose([A.HueSaturationValue(p=1.0)]),
    "channel_shuffle": A.Compose([A.ChannelShuffle(p=1.0)]),
    "blur": A.Compose([A.GaussianBlur(blur_limit=(5, 15), p=1.0)]),  # 연한 블러
    "strong_noise": A.Compose([A.GaussNoise(var_limit=(300.0, 500.0), p=1.0)]),
    "to_gray": A.Compose([A.ToGray(p=1.0)]),
    "gray2": A.Compose([A.ToGray(p=1.0)]),
    "brightness_contrast": A.Compose([A.RandomBrightnessContrast(p=1.0)]),  # 밝기 및 대비 조정
    "clahe": A.Compose([A.CLAHE(p=1.0)]),  # CLAHE 적용
    "solarize": A.Compose([A.Solarize(p=1.0)]),  # Solarize 효과
    "invert": A.Compose([A.InvertImg(p=1.0)]),  # 색상 반전
    "light_blur": A.Compose([A.MedianBlur(blur_limit=7, p=1.0)]),  # 연한 Median 블러
    "equalize": A.Compose([A.Equalize(p=1.0)])  # 히스토그램 평활화
}

# 디렉토리의 모든 이미지 파일에 대해 증강 작업 수행
for filename in os.listdir(image_directory):
    print(filename)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # 이미지 로드
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        # 각 증강 기법 적용 및 결과 저장
        for name, transform in transformations.items():
            transformed_image = transform(image=image)['image']
            output_image_name = f"{filename.split('.')[0]}_{name}.jpg"  # 이미지 이름 변경
            output_image_path = os.path.join(image_directory, output_image_name)
            cv2.imwrite(output_image_path, transformed_image)

            # 라벨 파일 경로 구성 및 복사
            label_filename = os.path.splitext(filename)[0] + '.txt'  # 원본 라벨 파일명
            label_path = os.path.join(label_directory, label_filename)
            if os.path.exists(label_path):  # 라벨 파일이 존재하는 경우에만 복사
                output_label_name = f"{filename.split('.')[0]}_{name}.txt"  # 라벨 파일명 변경
                output_label_path = os.path.join(label_directory, output_label_name)
                shutil.copy(label_path, output_label_path)

        # 여기에 이미지 및 라벨 파일 복사 추가
        for version in ['_v2', '_v3', 'v4']:
            # 이미지 복사
            new_image_name = f"{filename.split('.')[0]}{version}.jpg"
            new_image_path = os.path.join(image_directory, new_image_name)
            shutil.copy(image_path, new_image_path)
            
            # 라벨 파일 복사
            new_label_name = f"{filename.split('.')[0]}{version}.txt"
            new_label_path = os.path.join(label_directory, new_label_name)
            if os.path.exists(label_path):
                shutil.copy(label_path, new_label_path)