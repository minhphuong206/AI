import os
import cv2
import numpy as np

# --- 1. CẤU HÌNH ---
DATASET_DIR = "dataset"
# Chú ý: Tên trong danh sách này phải khớp TỪNG CHỮ với tên thư mục bạn đã tạo
CATEGORIES = ["nhan_vien_1", "nhan_vien_2","nhan_vien_3", "nguoi_la"]

X = [] # Danh sách chứa ma trận ảnh
y = [] # Danh sách chứa nhãn

print(" Đang tiến hành gom và đóng gói dữ liệu...")

# --- 2. QUÉT VÀ ĐỌC DỮ LIỆU ---
for category in CATEGORIES:
    # Lấy đường dẫn tới thư mục gốc của từng nhóm
    path = os.path.join(DATASET_DIR, category)
    # Lấy nhãn số (0, 1, hoặc 2) dựa vào vị trí trong danh sách CATEGORIES
    class_num = CATEGORIES.index(category) 
    
    # Dùng os.walk để lùng sục vào mọi thư mục con (truc_dien, nghieng_trai...)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                try:
                    img_path = os.path.join(root, file)
                    # Đọc ảnh thô (ảnh xám)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Chỉ lấy những ảnh đúng chuẩn 64x64
                    if img_array is not None and img_array.shape == (64, 64):
                        X.append(img_array)
                        y.append(class_num)
                except Exception as e:
                    pass # Bỏ qua nếu có ảnh bị lỗi file

# --- 3. TIỀN XỬ LÝ TOÁN HỌC CUỐI CÙNG ---
print(" Đang chuyển đổi sang định dạng AI...")

# Chuyển đổi list thành mảng Numpy Array
# Đồng thời chia cho 255.0 để Chuẩn hóa (Normalization) dữ liệu về khoảng 0 -> 1. 
# Việc này giúp AI học nhanh và chính xác hơn rất nhiều.
X = np.array(X).astype('float32') / 255.0 

# Định hình lại ma trận X cho phù hợp với mạng CNN.
# Định dạng yêu cầu: (Số lượng ảnh, Chiều cao, Chiều rộng, Số kênh màu)
# Vì là ảnh đen trắng nên Số kênh màu = 1
X = X.reshape(-1, 64, 64, 1)

y = np.array(y)

# --- 4. LƯU THÀNH FILE ---
print("-" * 30)
print(f" Tổng số ảnh hợp lệ đã đóng gói: {len(X)} ảnh")
print(f" Kích thước tập X (Dữ liệu): {X.shape}")
print(f" Kích thước tập y (Nhãn): {y.shape}")
print("-" * 30)

# Lưu lại thành file .npy để bước sau chỉ việc load lên train, không cần đọc lại ảnh nữa
np.save('X.npy', X)
np.save('y.npy', y)

print(" HOÀN TẤT! Đã xuất ra 2 file X.npy và y.npy tại thư mục gốc.")