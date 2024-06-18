from PIL import Image
import os
import shutil

# 지원하는 이미지 파일 확장자
supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize_image(image_path, label_path, images_output_dir, labels_output_dir):
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        resized = False  # Flag to check if image was resized

        # Resize conditions
        if (original_width > 1708 and original_height <= 960) or (original_height > 1708 and original_width <= 960):
            new_width = 1920 if original_width > original_height else int(original_width * (1920 / original_height))
            new_height = int(original_height * (1920 / original_width)) if original_width > original_height else 1920
            img = img.resize((new_width, new_height), Image.LANCZOS)
            resized = True
        elif original_width > 1708 or original_height > 1708:
            if original_width >= original_height:  # Landscape or square
                new_width = 1708
                new_height = int((original_height / original_width) * 1708)
            else:  # Portrait
                new_height = 1708
                new_width = int((original_width / original_height) * 1708)
            if new_width < 960:
                new_width = 960
            if new_height < 960:
                new_height = 960
            img = img.resize((new_width, new_height), Image.LANCZOS)
            resized = True
        
        # Original 저장
        original_image_output_path = os.path.join(images_output_dir, os.path.basename(image_path))
        img.save(original_image_output_path)
        print(f"Original image saved to {original_image_output_path}")

        original_label_output_path = os.path.join(labels_output_dir, os.path.basename(label_path))
        shutil.copy(label_path, original_label_output_path)
        print(f"Original label saved to {original_label_output_path}")

        # Save the resized image if it was resized
        if resized:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            resized_image_path = os.path.join(images_output_dir, f"{base_filename}_resized.jpg")
            img.save(resized_image_path)
            print(f"Resized image saved to {resized_image_path}")

            # Save the label file with the new filename
            resized_label_path = os.path.join(labels_output_dir, f"{base_filename}_resized.txt")
            shutil.copy(label_path, resized_label_path)
            print(f"Resized label saved to {resized_label_path}")
            
            return resized_image_path, resized_label_path
        else:
            return image_path, label_path

def split_image_and_labels(image_path, label_path, images_output_dir, labels_output_dir):
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        # Determine how to split the image
        split_vertical = original_width > 960
        split_horizontal = original_height > 960

        # If the image does not need to be split, copy the original image and label
        if not split_vertical and not split_horizontal:
            shutil.copy(image_path, os.path.join(images_output_dir, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(labels_output_dir, os.path.basename(label_path)))
            return

        # Split the image if needed
        crops = []
        if split_vertical:
            crops.extend([(0, 0, min(960, original_width), original_height),
                          (max(0, original_width - 960), 0, original_width, original_height)])
        if split_horizontal:
            crops.extend([(0, 0, original_width, min(960, original_height)),
                          (0, max(0, original_height - 960), original_width, original_height)])
        
        for i, (left, upper, right, lower) in enumerate(crops, start=1):
            cropped_img = img.crop((left, upper, right, lower)).convert('RGB')  # Ensure the image is in RGB
            cropped_img_path = os.path.join(images_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_part{i}.jpg")
            cropped_img.save(cropped_img_path)
            print(f"Saved split image to {cropped_img_path}")

            # Adjust and save labels for the cropped image
            with open(label_path, 'r') as f, open(os.path.join(labels_output_dir, f"{os.path.splitext(os.path.basename(cropped_img_path))[0]}.txt"), 'w') as out_f:
                for line in f:
                    parts = line.split()
                    class_id = int(parts[0])  # 클래스 인덱스를 정수로 변환
                    x_center, y_center, width, height = map(float, parts[1:])

                    bbox_left = (x_center - width / 2) * original_width
                    bbox_top = (y_center - height / 2) * original_height
                    bbox_right = (x_center + width / 2) * original_width
                    bbox_bottom = (y_center + height / 2) * original_height

                    # Check if the bounding box is within the cropped image
                    if bbox_right > left and bbox_left < right and bbox_bottom > upper and bbox_top < lower:
                        # Adjust coordinates and save
                        new_x_center = ((bbox_left + bbox_right) / 2 - left) / (right - left)
                        new_y_center = ((bbox_top + bbox_bottom) / 2 - upper) / (lower - upper)
                        new_width = (bbox_right - bbox_left) / (right - left)
                        new_height = (bbox_bottom - bbox_top) / (lower - upper)

                        # Format the numbers with 6 decimal places and write to the file
                        out_f.write(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")

def process_directory(image_dir, label_dir, output_dir):
    images_output_dir = os.path.join(output_dir, "images")
    labels_output_dir = os.path.join(output_dir, "labels")

    ensure_dir(images_output_dir)
    ensure_dir(labels_output_dir)

    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1].lower() in supported_extensions:
            image_path = os.path.join(image_dir, filename)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                # 여기에서 label_path도 함께 넘겨줘야 합니다.
                resized_image_path, resized_label_path = resize_image(image_path, label_path, images_output_dir, labels_output_dir)
                split_image_and_labels(resized_image_path, resized_label_path, images_output_dir, labels_output_dir)
            else:
                print(f"Label file for {filename} not found, skipping.")

# Paths
image_dir = 'classdata/train/images'
label_dir = 'classdata/train/labels'
output_dir = 'classdata/processtrain'

# Process all images and labels in the directory
process_directory(image_dir, label_dir, output_dir)