import glob
import os
import json
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image/dir")
ap.add_argument("-t", "--txt", type=str, required=True,
                help="path to txt/dir")
ap.add_argument("-c", "--classes", type=str, required=True,
                help="path to classes.txt")

args = vars(ap.parse_args())
path_to_img = args["image"]
path_to_txt = args['txt']
path_to_class = args['classes']
path_to_annotaitons = "class0518/annotations"

txt_list = sorted(glob.glob(f'{path_to_txt}/*.txt'))
img_full_list = glob.glob(f'{path_to_img}/*.jpeg') + \
                glob.glob(f'{path_to_img}/*.jpg')  + \
                glob.glob(f'{path_to_img}/*.png')

img_list = sorted(img_full_list)
class_names = open(path_to_class, 'r+').read().splitlines()
class_json = []
for i in range(len(class_names)):
    class_ = {
        "id": i,
        "name": class_names[i]
        }
    class_json.append(class_)
img_json_list = []
txt_json_list = []
c_img = 0
c_anot = 0
for txt, img in zip(txt_list, img_list):
    # print(txt)
    # Image
    folder_name, file_name = os.path.split(img)
    img_file = cv2.imread(img)
    height_n, width_n, depth_n = img_file.shape
    img_dict = {
            "id": c_img,
            "license": 1,
            "file_name": file_name,
            "height": height_n,
            "width": width_n,
        }
    img_json_list.append(img_dict)

    # Annotation
    lines = open(txt, 'r+').read().splitlines()
    obj_list = []
    class_list = []
    for line in lines:
        class_index, x_center, y_center, width, height = line.split()
        # 문자열을 float으로 변환
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)

        # 계산된 값을 사용하여 xmax, xmin, ymax, ymin 계산
        xmax = (x_center * width_n) + (width * width_n / 2.0)
        xmin = (x_center * width_n) - (width * width_n / 2.0)
        ymax = (y_center * height_n) + (height * height_n / 2.0)
        ymin = (y_center * height_n) - (height * height_n / 2.0)

        # 결과를 정수로 변환
        xmax = int(xmax)
        xmin = int(xmin)
        ymax = int(ymax)
        ymin = int(ymin)
        
        # width_anot과 height_anot 계산
        width_anot = xmax - xmin  # float 변환 제거
        height_anot = ymax - ymin  # float 변환 제거
        # print(xmax, xmin, ymax, ymin, width_anot, height_anot)
        # Annotations Dict
        annotation = {
            "id": c_anot,
            "image_id": c_img,
            "category_id": int(class_index),
            "bbox": [
                xmin,
                ymin,
                width_anot,
                height_anot
            ],
            "area": width_anot*height_anot,
            "segmentation": [],
            "iscrowd": 0
        }
        txt_json_list.append(annotation)
        c_anot += 1
    c_img += 1
full_json = {
    "categories": class_json,
    "images": img_json_list,
    "annotations": txt_json_list
}
json_object = json.dumps(full_json, indent=3)
with open(f"{path_to_annotaitons}/{path_to_img.split('/')[-2]}.json", "w") as outfile:
    outfile.write(json_object)
print(f"[INFO] Saved JSON: {path_to_annotaitons}/{path_to_img.split('/')[-2]}.json")