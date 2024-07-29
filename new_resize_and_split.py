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
        if original_width > 1920 or original_height > 1920:
            if original_width > original_height:
                new_width = 1920
                new_height = int(original_height * (1920 / original_width))
            else:
                new_height = 1920
                new_width = int(original_width * (1920 / original_height))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            resized = True

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        original_image_output_path = os.path.join(images_output_dir, f"{base_filename}.jpg")
        original_label_output_path = os.path.join(labels_output_dir, f"{base_filename}.txt")

        # Save original or resized image
        if resized:
            resized_image_path = os.path.join(images_output_dir, f"{base_filename}_sizeup.jpg")
            # Convert image to RGB if saving as JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(resized_image_path)
            resized_label_path = os.path.join(labels_output_dir, f"{base_filename}_sizeup.txt")
            shutil.copy(label_path, resized_label_path)
            return resized_image_path, resized_label_path
        else:
            # Convert image to RGB if saving as JPEG
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(original_image_output_path)
            shutil.copy(label_path, original_label_output_path)
            return original_image_output_path, original_label_output_path

def split_image_and_labels(image_path, label_path, images_output_dir, labels_output_dir):
    with Image.open(image_path) as img:
        original_width, original_height = img.size

        # Check if splitting is necessary
        if original_width <= 960 and original_height <= 960:
            destination_image_path = os.path.join(images_output_dir, os.path.basename(image_path))
            if destination_image_path != image_path:
                shutil.copy(image_path, destination_image_path)
            destination_label_path = os.path.join(labels_output_dir, os.path.basename(label_path))
            if destination_label_path != label_path:
                shutil.copy(label_path, destination_label_path)
            return
        
        # Determine cropping areas based on image dimensions
        crop_areas = []

        if original_width > 960 and original_height > 960:
            # Both dimensions are greater than 960
            widths = [0, original_width // 2 - 480, original_width - 960]
            heights = [0, original_height // 2 - 480, original_height - 960]
            for i, left in enumerate(widths):
                for j, upper in enumerate(heights):
                    right = min(left + 960, original_width)
                    lower = min(upper + 960, original_height)
                    crop_areas.append((left, upper, right, lower))
        elif original_width > 960:
            # Only width is greater than 960
            crop_areas = [
                (0, 0, 960, original_height),
                (original_width // 2 - 480, 0, original_width // 2 + 480, original_height),
                (original_width - 960, 0, original_width, original_height)
            ]
        elif original_height > 960:
            # Only height is greater than 960
            crop_areas = [
                (0, 0, original_width, 960),
                (0, original_height // 2 - 480, original_width, original_height // 2 + 480),
                (0, original_height - 960, original_width, original_height)
            ]

        # Crop the image into parts and adjust labels
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        for idx, (left, upper, right, lower) in enumerate(crop_areas, start=1):
            part_filename = f"{base_filename}_part{idx}"
            cropped_img_path = os.path.join(images_output_dir, f"{part_filename}.jpg")
            cropped_img = img.crop((left, upper, right, lower))
            # Convert image to RGB if saving as JPEG
            if cropped_img.mode != 'RGB':
                cropped_img = cropped_img.convert('RGB')
            cropped_img.save(cropped_img_path)

            # Adjust and save labels for the cropped image
            with open(label_path, 'r') as f, open(os.path.join(labels_output_dir, f"{part_filename}.txt"), 'w') as out_f:
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
                resized_image_path, resized_label_path = resize_image(image_path, label_path, images_output_dir, labels_output_dir)
                split_image_and_labels(resized_image_path, resized_label_path, images_output_dir, labels_output_dir)
            else:
                print(f"Label file for {filename} not found, skipping.")

# Paths
filename = "lawimage1"
image_dir = f'{filename}/train/images'
label_dir = f'{filename}/train/labels'
output_dir = f'{filename}/processtrain'

# Process all images and labels in the directory
process_directory(image_dir, label_dir, output_dir)