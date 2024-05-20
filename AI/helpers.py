import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def show_images(images, labels, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    # Duyệt qua 8 hình ảnh đầu tiên trong mảng NumPy và hiển thị chúng
    for i in range(rows * cols):
        # Lấy hình ảnh từ mảng NumPy
        image = images[i]  # image_array là mảng chứa các hình ảnh của bạn
        
        # Tính toán chỉ số hàng và cột cho subplot
        row_index = i // cols
        col_index = i % cols
        
        # Hiển thị hình ảnh
        axes[row_index, col_index].imshow(image,cmap='gray')
        axes[row_index, col_index].axis('off')  # Tắt các trục
        axes[row_index, col_index].set_title( str(labels[i]))
        
    # Hiển thị tất cả các subplot
    plt.tight_layout()
    plt.show()


def create_full_contrast(input):
    min_val = np.min(input)
    max_val = np.max(input)
    if min_val == max_val:
        return np.zeros_like(input, dtype=np.uint8)
    scale = 255/(max_val - min_val)
    normalized = np.round(((input - min_val) * scale))
    return normalized

def bgr_2_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def sharpened_filter(image):
    # Áp dụng bộ lọc Laplacian để tìm biên cạnh
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Tạo hình ảnh sắc nét bằng cách kết hợp hình ảnh gốc với biên cạnh
    sharpened_image = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0, dtype=cv2.CV_8U)
    return sharpened_image

def equalizeHist_filter(image):
    return cv2.equalizeHist(image)

def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def load_dataset_from_directory(directory, shuffle=False, shape = (20,20) ):
    images = []
    labels = []

    # Duyệt qua từng thư mục con trong directory
    current_label = 0
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        # Kiểm tra xem subdir có phải là thư mục không
        if os.path.isdir(subdir_path):
            current_label = int(subdir)
            # Duyệt qua các tập tin hình ảnh trong thư mục con
            for filename in os.listdir(subdir_path):
                # Xác định đường dẫn đầy đủ của hình ảnh
                img_path = os.path.join(subdir_path, filename)
                # Kiểm tra xem tập tin có phải là hình ảnh không
                if os.path.isfile(img_path) and (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
                    # Đọc hình ảnh bằng OpenCV
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, shape)
                    # Thêm hình ảnh vào danh sách images
                    images.append(image)
                    # Thêm nhãn tương ứng vào danh sách labels
                    labels.append(current_label)

    # Chuyển danh sách thành mảng NumPy
    images = np.array(images)
    labels = np.array(labels)

    if shuffle:
        indices = np.random.permutation(len(images))
        images = images[indices]
        labels = labels[indices]

    return images, labels

def hex_to_str_array(hex_data):
  hex_array = []
  for i, val in enumerate(hex_data) :
    # Construct string from hex
    hex_str = format(val, '#04x')
    hex_array.append(hex_str)
  return hex_array


