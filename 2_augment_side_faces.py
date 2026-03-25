import cv2
import os
import numpy as np

# --- 1. CẤU HÌNH (QUAN TRỌNG: Cần chạy 2 lần cho 2 góc) ---

# --- LẦN 1: Cấu hình cho góc NGHIÊNG TRÁI ---
#INPUT_SEED_IMAGE = "dataset/nhan_vien_3/nghieng_trai/goc_trai.jpg" 
#OUTPUT_DIR = "dataset/nhan_vien_3/nghieng_trai/"

# --- LẦN 2: Bỏ comment 2 dòng này khi chạy lần 2 ---
#INPUT_SEED_IMAGE = "dataset/nhan_vien_3/nghieng_phai/goc_phai.jpg"
#OUTPUT_DIR = "dataset/nhan_vien_3/nghieng_phai/"
#INPUT_SEED_IMAGE = "dataset/nhan_vien_3/goc_duoi/goc_xuong.jpg"
#OUTPUT_DIR = "dataset/nhan_vien_3/goc_duoi/"
#INPUT_SEED_IMAGE = "dataset/nhan_vien_3/goc_tren/goc_tren.jpg"
#OUTPUT_DIR = "dataset/nhan_vien_3/goc_tren/"
INPUT_SEED_IMAGE = "dataset/nguoi_la/nghieng_phai/goc_phai.jpg"
OUTPUT_DIR = "dataset/nguoi_la/nghieng_phai/"
# --- Thông số chung ---
IMAGE_SIZE = (64, 64) # Kích thước ảnh thô cuối cùng
DESIRED_COUNT = 30 # Số lượng ảnh muốn tạo ra

# --- 2. TIỀN XỬ LÝ ẢNH "HẠT GIỐNG" ---
print(f" Đang xử lý ảnh hạt giống: {INPUT_SEED_IMAGE}...")

if not os.path.exists(INPUT_SEED_IMAGE):
    print(f" Lỗi: Không tìm thấy file {INPUT_SEED_IMAGE}. Hãy chắc chắn bạn đã cắt và lưu đúng chỗ.")
    exit()

# Đọc ảnh màu bạn vừa cắt tay
img = cv2.imread(INPUT_SEED_IMAGE)

# 2.1. Đồng nhất kích thước về 64x64 NGAY LẬP TỨC
# Điều này giúp các phép biến đổi sau đó (xoay, nhiễu) đồng nhất trên khung 64x64
resized_img = cv2.resize(img, IMAGE_SIZE)

# 2.2. Chuyển ảnh sang ảnh xám (Grayscale)
gray_base_face = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Lưu khuôn mặt thô gốc (đen trắng, 64x64) đầu tiên
base_filename = "face_manual_base.jpg"
cv2.imwrite(os.path.join(OUTPUT_DIR, base_filename), gray_base_face)
print(f" Đã lưu khuôn mặt thô 'gốc' vào: {OUTPUT_DIR}/{base_filename}")


# --- 3. TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION) ---
print(f" Đang tạo thêm {DESIRED_COUNT-1} ảnh nghiêng biến thể bằng Data Augmentation...")

count = 1
while count < DESIRED_COUNT:
    # --- Biến thể 1: Xoay ảnh (Rotation) ---
    # Góc nghiêng cần hạn chế xoay quá nhiều, khoảng +/- 10 độ là vừa
    angle = np.random.uniform(-10, 10) 
    M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2), angle, 1)
    rotated_img = cv2.warpAffine(gray_base_face, M, IMAGE_SIZE)

    # --- Biến thể 2: Thay đổi độ sáng ngẫu nhiên (Brightness) ---
    # Phù hợp với môi trường tối của bạn
    brightness_factor = np.random.uniform(0.7, 1.3) 
    brightened_img = cv2.convertScaleAbs(rotated_img, alpha=brightness_factor, beta=0)

    # --- Biến thể 3: Thêm nhiễu ngẫu nhiên chuẩn xác (Noise) ---
    # Sử dụng kỹ thuật cộng số thực (float) và clip để tránh lỗi đốm trắng
    noise = np.random.normal(0, 7, brightened_img.shape) # Nhiễu nhẹ hơn góc thẳng một chút
    noisy_img = brightened_img.astype(np.float32) + noise
    
    # Chốt các giá trị lại trong khoảng 0-255 rồi chuyển về uint8
    final_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Lưu ảnh biến thể mới
    filename = f"face_manual_aug_{count}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
    
    count += 1

print(f" Hoàn tất góc nghiêng! Hãy kiểm tra thư mục {OUTPUT_DIR} để thấy 30 tấm ảnh thô đã sẵn sàng.")