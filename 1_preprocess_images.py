import cv2
import os
import numpy as np

# --- 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THÔNG SỐ ---
# Đường dẫn ảnh gốc bạn vừa tìm được
INPUT_IMAGE = "nguoi_la_10.jpg" 
#INPUT_IMAGE = "nhan_vien_3.jpg" 

# Thư mục để lưu ảnh "thô" sau khi xử lý
#OUTPUT_DIR = "dataset/nhan_vien_1/truc_dien/"
OUTPUT_DIR = "dataset/nguoi_la/"
# Kích thước ảnh thô cuối cùng (ví dụ: 64x64 pixel)
IMAGE_SIZE = (64, 64) 

# Số lượng ảnh chúng ta muốn tạo ra từ Data Augmentation (Tăng cường dữ liệu)
DESIRED_COUNT = 30 

# Load bộ phân loại khuôn mặt mặc định của OpenCV (không phải AI train sẵn)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. TẠO THƯ MỤC ĐẦU RA NẾU CHƯA CÓ ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f" Đã tạo thư mục lưu trữ: {OUTPUT_DIR}")

# --- 3. TIỀN XỬ LÝ: CẮT, CHUYỂN XÁM, RESIZE ---
print(" Đang đọc ảnh gốc và trích xuất khuôn mặt...")
# Đọc ảnh gốc
img = cv2.imread(INPUT_IMAGE)
# Chuyển ảnh màu sang ảnh xám (Grayscale)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Nhận diện các khuôn mặt trong ảnh
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print(" Lỗi: Không nhận diện được khuôn mặt nào trong ảnh gốc. Hãy kiểm tra lại ảnh.")
    exit()

print(f" Đã tìm thấy {len(faces)} khuôn mặt khác nhau.")

# Trong bức ảnh ghép của bạn, chúng ta sẽ chỉ lấy 1 khuôn mặt NHÌN THẲNG đầu tiên để test.
# Chúng ta sẽ tìm khuôn mặt nằm ở phía trên bên trái nhất.
# Sắp xếp các khuôn mặt theo vị trí y (dòng) rồi x (cột)
sorted_faces = sorted(faces, key=lambda f: (f[1], f[0]))
(x, y, w, h) = sorted_faces[0] # Lấy khuôn mặt đầu tiên (góc trên bên trái)

# Cắt ảnh lấy đúng khuôn mặt
face_roi = gray_img[y:y+h, x:x+w]

# Đồng nhất kích thước về 64x64
final_base_face = cv2.resize(face_roi, IMAGE_SIZE)

# Lưu khuôn mặt gốc làm ảnh thô đầu tiên
cv2.imwrite(os.path.join(OUTPUT_DIR, "face_original.jpg"), final_base_face)
print(f" Đã lưu khuôn mặt 'gốc' đầu tiên vào: {OUTPUT_DIR}/face_original.jpg")


# --- 4. TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION) ---
print(f" Đang tạo thêm {DESIRED_COUNT-1} ảnh khác nhau bằng kỹ thuật Data Augmentation...")

count = 1
while count < DESIRED_COUNT:
    # --- Biến thể 1: Xoay ảnh (Rotation) ---
    angle = np.random.uniform(-15, 15) # Xoay ngẫu nhiên từ -15 đến +15 độ
    M = cv2.getRotationMatrix2D((IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2), angle, 1)
    rotated_img = cv2.warpAffine(final_base_face, M, IMAGE_SIZE)

    # --- Biến thể 2: Thay đổi độ sáng ngẫu nhiên (Brightness) ---
    # Mình thu hẹp khoảng sáng/tối lại một chút (0.7 đến 1.3) để mặt không bị cháy sáng
    brightness_factor = np.random.uniform(0.7, 1.3) 
    brightened_img = cv2.convertScaleAbs(rotated_img, alpha=brightness_factor, beta=0)

    # --- Biến thể 3: Thêm nhiễu ngẫu nhiên chuẩn xác (Noise) ---
    # Tạo nhiễu và cộng vào ảnh dưới dạng số thực (float) để không bị lỗi số âm
    noise = np.random.normal(0, 8, brightened_img.shape)
    noisy_img = brightened_img.astype(np.float32) + noise
    
    # Chốt các giá trị lại trong khoảng 0-255 (chuẩn của ảnh) rồi mới xuất ra
    final_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Lưu ảnh biến thể mới
    filename = f"face_aug_{count}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), final_img)
    
    count += 1

    

print(f" Hoàn tất! Bạn có thể kiểm tra thư mục {OUTPUT_DIR} để thấy 30 tấm ảnh thô đã sẵn sàng.")