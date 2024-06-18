# ##### 계속 실행하다가 확인
# import cv2
# import torch
# import numpy as np
# import random
# import time
# import os
# from threading import Thread
# import pandas as pd
# from super_gradients.training import models
# import pyttsx3

# # 음성 경고를 위한 함수 설정
# def speak_warning(message):
#     engine = pyttsx3.init()
#     engine.say(message)
#     engine.runAndWait()

# # 설정 변수
# num_classes = 28
# model_type = 'yolo_nas_l'
# weight_path = 'ckpt_best_50.pth'
# confidence_threshold = 0.45
# save_images = True  # 이미지를 저장할지 여부
# hide_window = False  # 추론 윈도우를 숨길지 여부

# # Ensure output directory exists
# os.makedirs('runs/detect', exist_ok=True)
# stats_file_path = os.path.join('runs', 'detect', 'webcam_stats.csv')

# # If the statistics file exists, remove it before starting
# if os.path.exists(stats_file_path):
#     os.remove(stats_file_path)

# # Initialize the data list for thread-safe writing to the stats list
# stats_data = []
# csv_columns = ['Frame', 'Label', 'Count', 'Average Confidence', 'Total Count']

# def save_stats():
#     """Save the current statistics from stats_data to a CSV file."""
#     global stats_data
#     if stats_data:
#         pd.DataFrame(stats_data, columns=csv_columns).to_csv(stats_file_path, index=False, mode='a', header=not os.path.exists(stats_file_path))
#         stats_data = []

# def save_image(img, frame_count):
#     """Save the current frame as an image file."""
#     img_filename = os.path.join('runs', 'detect', f'frame_{frame_count}.jpg')
#     cv2.imwrite(img_filename, img)
#     print(f'[INFO] Saved image: {img_filename}')

# # Function to draw bounding boxes on image
# def plot_one_box(x, img, color=None, label=None, line_thickness=3):
#     tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
#         cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# # Load the YOLO-NAS model
# model = models.get(model_type, num_classes=num_classes, checkpoint_path=weight_path)
# model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# class_names = model.predict(np.zeros((1,1,3)), conf=confidence_threshold)._images_prediction_lst[0].class_names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# # Open webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# frame_count = 0
# capture_frame_interval = 50  # Adjust to capture every 100 frames
# global_timer = time.time()

# while True:
#     success, img = cap.read()
#     if not success:
#         print('[INFO] Failed to capture image from webcam...')
#         break

#     frame_count += 1

#     if frame_count % capture_frame_interval == 0:
#         # Perform full detection with statistics and saving images
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         preds = model.predict(img_rgb, conf=confidence_threshold)._images_prediction_lst[0]
#         dp = preds.prediction
#         bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

#         total_count_per_frame = 0
#         for box, cnf, label in zip(bboxes, confs, labels):
#             class_name = class_names[int(label)]
#             plot_one_box(box[:4], img, label=f'{class_name} {cnf:.2f}', color=colors[label])
#             stats_data.append([frame_count, class_name, 1, cnf, 1])
#             total_count_per_frame += 1

#         # Save stats and images every 100 frames
#         Thread(target=save_stats).start()
#         if save_images:
#             Thread(target=save_image, args=(img, frame_count)).start()
        
#         # Check if the last set of stats contain 'sleep'
#         if any(sd[1] == 'sleep' for sd in stats_data):
#             print('[WARNING] 위반카드 Detected!')
#             Thread(target=speak_warning, args=('일어나세요! 위반카드',)).start()

#     else:
#         # Perform only detection without saving statistics
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         preds = model.predict(img_rgb, conf=confidence_threshold)._images_prediction_lst[0]
#         dp = preds.prediction
#         bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

#         for box, cnf, label in zip(bboxes, confs, labels):
#             class_name = class_names[int(label)]
#             plot_one_box(box[:4], img, label=f'{class_name} {cnf:.2f}', color=colors[label])

#     cv2.putText(img, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

#     if not hide_window:
#         cv2.imshow('Inference', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# print(f'[INFO] Completed in {(time.time() - global_timer) / 60:.2f} minutes.')
# cap.release()
# if not hide_window:
#     cv2.destroyAllWindows()


import cv2
import torch
import numpy as np
import random
import time
import os
from threading import Thread
import pandas as pd
from super_gradients.training import models
import pyttsx3
from collections import Counter

# 음성 경고를 위한 함수 설정
def speak_warning(message):
    engine = pyttsx3.init()
    engine.say(message)
    engine.runAndWait()

# 설정 변수
num_classes = 28
model_type = 'yolo_nas_l'
weight_path = 'ckpt_best_50.pth'
confidence_threshold = 0.45
save_images = True  # 이미지를 저장할지 여부
hide_window = False  # 추론 윈도우를 숨길지 여부

# Ensure output directory exists
os.makedirs('runs/detect', exist_ok=True)
stats_file_path = os.path.join('runs', 'detect', 'webcam_stats.csv')

# Path for concentration data
concentration_data_path = 'runs/detect/concentrate_data.csv'

# If the statistics file exists, remove it before starting
if os.path.exists(stats_file_path):
    os.remove(stats_file_path)

# Initialize the data list for thread-safe writing to the stats list
stats_data = []
csv_columns = ['Frame', 'Label', 'Count', 'Average Confidence', 'Total Count', 'Average Concentration']

# Initialize the data list for concentration measurements
concentration_measurements = []
concentration_columns = ['Time', 'Concentrate_rate']

def save_stats():
    """Save the current statistics from stats_data to a CSV file."""
    global stats_data
    if stats_data:
        pd.DataFrame(stats_data, columns=csv_columns).to_csv(stats_file_path, index=False, mode='a', header=not os.path.exists(stats_file_path))
        stats_data = []

def save_concentration_measurements():
    """Save the current concentration measurements to a CSV file."""
    global concentration_measurements
    if concentration_measurements:
        pd.DataFrame(concentration_measurements, columns=concentration_columns).to_csv(concentration_data_path, index=False, mode='a', header=not os.path.exists(concentration_data_path))
        concentration_measurements = []

def save_image(img, frame_count):
    """Save the current frame as an image file."""
    img_filename = os.path.join('runs', 'detect', f'frame_{frame_count}.jpg')
    cv2.imwrite(img_filename, img)
    print(f'[INFO] Saved image: {img_filename}')

# Function to draw bounding boxes on image
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

# Load the YOLO-NAS model
model = models.get(model_type, num_classes=num_classes, checkpoint_path=weight_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=confidence_threshold)._images_prediction_lst[0].class_names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0
capture_frame_interval = 50  # Adjust to capture every 50 frames
global_timer = time.time()

# Define concentration scores
concentration_scores = {
    'seeteacher': 1.0,
    'seemonitor': 1.0,
    'seebook':0.9,
    'sleep': 0.0,
    'seeother': 0.2,
    'somethingelse': 0.1,
    'drinkwater' : 0.5
}

# Initialize a timer for concentration measurement
last_measurement_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed to capture image from webcam...')
        break

    frame_count += 1

    # Perform detection
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = model.predict(img_rgb, conf=confidence_threshold)._images_prediction_lst[0]
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

    label_counts = Counter()
    total_concentration = 0
    for box, cnf, label in zip(bboxes, confs, labels):
        class_name = class_names[int(label)]
        plot_one_box(box[:4], img, label=f'{class_name} {cnf:.2f}', color=colors[label])
        concentration = concentration_scores.get(class_name, 0)
        label_counts[class_name] += 1
        total_concentration += concentration

    total_count_per_frame = sum(label_counts.values())
    average_concentration = total_concentration / total_count_per_frame if total_count_per_frame > 0 else 0

    # Display concentration on the frame
    cv2.putText(img, f'Concentration: {average_concentration:.2f}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    current_time = time.time()
    if current_time - last_measurement_time >= 1.0:
        # Record the concentration measurement every second
        measurement_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        concentration_measurements.append([measurement_time, average_concentration])
        last_measurement_time = current_time

        # Save the current batch of concentration measurements
        Thread(target=save_concentration_measurements).start()

    if frame_count % capture_frame_interval == 0:
        # Save statistics and potentially images
        for label, count in label_counts.items():
            stats_data.append([frame_count, label, count, np.mean([confs[labels == class_names.index(label)]]), count, concentration_scores.get(label, 0)])

        Thread(target=save_stats).start()
        if save_images:
            Thread(target=save_image, args=(img, frame_count)).start()

        # Check warnings every 50 frames
        sleep_detected = label_counts['sleep'] >= 2
        somethingelse_or_seeother_detected = label_counts['somethingelse'] >= 2 or label_counts['seeother'] >= 2

        if sleep_detected:
            print('[WARNING] 자지말고 일어나세요! 위반카드')
            Thread(target=speak_warning, args=('자지말고 일어나세요! 위반카드',)).start()
        elif somethingelse_or_seeother_detected:
            print('[WARNING] 집중하세요! 위반카드')
            Thread(target=speak_warning, args=('집중하세요! 위반카드',)).start()

        if average_concentration < 0.5:  # Adjust threshold as needed
            print('[WARNING] 집중도가 낮습니다! 경고!')
            Thread(target=speak_warning, args=('집중도가 낮습니다! 경고!',)).start()

    cv2.putText(img, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    if not hide_window:
        cv2.imshow('Inference', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print(f'[INFO] Completed in {(time.time() - global_timer) / 60:.2f} minutes.')
cap.release()
if not hide_window:
    cv2.destroyAllWindows()