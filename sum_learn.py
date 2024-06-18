import os
import shutil

# 클래스 파일의 경로를 정의합니다.
classes1_path = 'class0511/classes.txt'
classes2_path = '0515addition/classes.txt'
base1_path = 'class0511'
base2_path = '0515addition'

# 라벨 변경할 경로
labels2_dirs = [
    '0515addition/train/labels',
    '0515addition/valid/labels'
]
# 복사할 폴더 목록을 정의합니다.
folders = ['train', 'valid', 'test']

# newData1의 classes.txt 파일을 읽습니다.
with open(classes1_path, 'r') as file:
    classes1 = file.read().splitlines()
classes1_count = len(classes1)
print(classes1_count)


# newData2의 classes.txt 파일을 읽습니다.
with open(classes2_path, 'r') as file:
    classes2 = file.read().splitlines()
classes2_count = len(classes2)
print(classes2_count)
# newData2에만 있는 클래스를 찾아서 newData1의 classes.txt에 추가합니다.
unique_classes2 = [cls for cls in classes2 if cls not in classes1]
classes1.extend(unique_classes2)
print('count :',len(classes1) )
# 수정된 classes.txt를 newData1에 저장합니다.
with open(classes1_path, 'w') as file:
    for cls in classes1:
        file.write(f"{cls}\n")

# newData2의 라벨 파일을 수정합니다.
for dir in labels2_dirs:
    if os.path.exists(dir):
        for file in os.listdir(dir):
            path = os.path.join(dir, file)

            # 각 라벨 파일을 읽고, 첫 번째 값을 수정합니다.
            with open(path, 'r') as f:
                lines = f.readlines()
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    class_index = int(parts[0])
                    class_name = classes2[class_index]
                    if class_name in classes1:
                        # 이미 존재하는 클래스의 경우, 원래 인덱스 사용
                        new_index = classes1.index(class_name)
                    else:
                        # 새로운 클래스의 경우, 인덱스 업데이트
                        new_index = class_index + classes1_count
                    updated_lines.append(f"{new_index} {' '.join(parts[1:])}\n")

            # 수정된 내용으로 라벨 파일을 다시 씁니다.
            with open(path, 'w') as f:
                f.writelines(updated_lines)
    else:
        print(f"디렉토리가 존재하지 않습니다: {dir}")

# newData2의 images와 labels를 newData1로 복사합니다.
for folder in folders:
    img2_path = os.path.join(base2_path, folder, 'images')
    lbl2_path = os.path.join(base2_path, folder, 'labels')
    
    img1_path = os.path.join(base1_path, folder, 'images')
    lbl1_path = os.path.join(base1_path, folder, 'labels')

    # images 폴더 내용을 복사합니다.
    if os.path.exists(img2_path):
        for item in os.listdir(img2_path):
            src = os.path.join(img2_path, item)
            dst = os.path.join(img1_path, item)
            if not os.path.exists(dst):  # 파일명 충돌 방지
                shutil.copy(src, dst)

    # labels 폴더 내용을 복사합니다.
    if os.path.exists(lbl2_path):
        for item in os.listdir(lbl2_path):
            src = os.path.join(lbl2_path, item)
            dst = os.path.join(lbl1_path, item)
            if not os.path.exists(dst):  # 파일명 충돌 방지
                shutil.copy(src, dst)
