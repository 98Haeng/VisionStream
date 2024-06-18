# import cv2
# import torch
# from super_gradients.training  import models
# from super_gradients.common.object_names import Models

# #models = models.get(Models.YOLO_NAS_l,pretrained_w)
# model = models.get(
#     'yolo_nas_l',
#     num_classes=28, 
#     checkpoint_path='ckpt_best_50.pth'
# )

# model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# outputs = model.predict_webcam()
# # print(outputs)
# outputs.show()
# print('output : ',outputs)
# models.convert_to_onnx(model = model, input_shape = (3,640,640), out_path = "yolo_nas_l.onnx")


##### 100에포크마다
# import cv2
# import torch
# import numpy as np
# import random
# import time
# from super_gradients.training import models
# import pandas as pd
# import os
# from threading import Thread

# # 설정 변수
# num_classes = 28
# model_type = 'yolo_nas_l'
# weight_path = 'ckpt_best_50.pth'
# confidence_threshold = 0.25
# save_images = True  # 이미지를 저장할지 여부
# hide_window = False  # 추론 윈도우를 숨길지 여부

# # Ensure output directory exists
# os.makedirs('runs/detect', exist_ok=True)
# stats_file_path = os.path.join('runs', 'detect', 'webcam_stats.csv')

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
# capture_frame_interval = 100  # Adjust to capture every 100 frames for stats and images
# global_timer = time.time()

# while True:
#     success, img = cap.read()
#     if not success:
#         print('[INFO] Failed to capture image from webcam...')
#         break

#     frame_count += 1
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     preds = model.predict(img_rgb, conf=confidence_threshold)._images_prediction_lst[0]
#     dp = preds.prediction
#     bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

#     # Draw bounding boxes on every frame
#     for box, cnf, label in zip(bboxes, confs, labels):
#         class_name = class_names[int(label)]
#         plot_one_box(box[:4], img, label=f'{class_name} {cnf:.2f}', color=colors[label])
#         if frame_count % capture_frame_interval == 0:
#             stats_data.append([frame_count, class_name, 1, cnf, 1])

#     # Every capture_frame_interval frames, save stats and images
#     if frame_count % capture_frame_interval == 0:
#         Thread(target=save_stats).start()
#         if save_images:
#             Thread(target=save_image, args=(img, frame_count)).start()

#     # Show the image in the window
#     cv2.putText(img, f'Frame: {frame_count}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

#     if not hide_window:
#         cv2.imshow('Inference', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# print(f'[INFO] Completed in {(time.time() - global_timer) / 60:.2f} minutes.')
# cap.release()
# if not hide_window:
#     cv2.destroyAllWindows()