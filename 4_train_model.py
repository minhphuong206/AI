import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

print(" Đang tải dữ liệu...")
# 1. TẢI DỮ LIỆU ĐÃ ĐÓNG GÓI
X = np.load('X.npy')
y = np.load('y.npy')

# 2. CHIA TẬP DỮ LIỆU (80% để học, 20% để thi thử)
# Việc này giúp đánh giá xem AI học thực sự hay chỉ "học vẹt"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f" Số ảnh dùng để học: {len(X_train)}")
print(f" Số ảnh dùng để thi thử: {len(X_test)}")

# 3. XÂY DỰNG KIẾN TRÚC MẠNG NƠ-RON (Mô hình CNN từ con số 0)
print(" Đang khởi tạo bộ não AI (CNN)...")
model = Sequential([
    # Lớp Tích chập 1: Quét các đường nét cơ bản (cạnh, góc)
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)), # Thu nhỏ để tập trung vào đặc trưng chính
    
    # Lớp Tích chập 2: Quét các bộ phận phức tạp hơn (mắt, mũi)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Duỗi thẳng ma trận ảnh thành 1 mảng 1 chiều
    Flatten(),
    
    # Lớp ẩn: Nơ-ron tư duy
    Dense(128, activation='relu'),
    Dropout(0.5), # Kỹ thuật "quên" ngẫu nhiên để chống học vẹt (Overfitting)
    
    # Lớp Đầu ra (Output): Đưa ra quyết định
    # Số 3 tương ứng với 3 class (nhan_vien_1, nhan_vien_2, nguoi_la)
    Dense(4, activation='softmax') 
])

# 4. BIÊN DỊCH MÔ HÌNH (Thiết lập thuật toán học)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 5. BẮT ĐẦU QUÁ TRÌNH HỌC (TRAINING)
print("-" * 40)
print(" BẮT ĐẦU HUẤN LUYỆN (TRAINING)...")
print("-" * 40)
# epochs=20 nghĩa là cho AI học đi học lại tập dữ liệu 20 lần
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# 6. LƯU BỘ NÃO ĐÃ HỌC XONG
model.save('model_chamcong.h5')
print("-" * 40)
print(" HOÀN TẤT! Đã lưu não bộ AI thành file 'model_chamcong.h5'")