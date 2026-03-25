import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- 1. TẢI BỘ NÃO VÀ CẤU HÌNH ---
print(" Đang thức tỉnh AI...")
model = load_model('model_chamcong.h5')

# Tên các nhãn (Phải đúng thứ tự 0, 1, 2, 3 như lúc train)
CATEGORIES = ["Nhan vien 1", "Nhan vien 2","nhan_vien_3","Nguoi la"]

# Tải công cụ tìm khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. MỞ CAMERA ---
cap = cv2.VideoCapture(0) # Số 0 thường là webcam mặc định của laptop

if not cap.isOpened():
    print(" Lỗi: Không thể mở camera.")
    exit()

print(" Camera đã sẵn sàng! Bấm phím 'q' trên cửa sổ camera để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật ngược ảnh camera cho giống soi gương (tùy chọn)
    frame = cv2.flip(frame, 1)

    # Chuyển ảnh sang xám để tìm khuôn mặt nhanh hơn
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   # Tìm các khuôn mặt đang xuất hiện trong camera
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Cắt khuôn mặt ra
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # --- TIỀN XỬ LÝ (Phải giống y hệt lúc train) ---
        resized_face = cv2.resize(face_roi, (64, 64))
        normalized_face = resized_face.astype('float32') / 255.0
        # Định hình lại thành (1, 64, 64, 1) vì AI nhận một "danh sách" chứa 1 bức ảnh
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

        # --- AI DỰ ĐOÁN ---
        # Tạm tắt thông báo in ra terminal liên tục để camera không bị giật
        predictions = model.predict(reshaped_face, verbose=0) 
        
        class_index = np.argmax(predictions) # Lấy vị trí có xác suất cao nhất (0, 1, hoặc 2)
        confidence = np.max(predictions) # Lấy độ tự tin (%)

      # --- HIỂN THỊ LÊN MÀN HÌNH ---
        # ĐẶT NGƯỠNG TỰ TIN (Ví dụ: 85%)
        NGUONG_TU_TIN = 0.85 
        
        if confidence < NGUONG_TU_TIN:
            # Nếu độ tự tin dưới 85%, dứt khoát coi là người lạ dù AI đoán là ai
            label = f"Khong xac dinh ({confidence*100:.1f}%)"
            color = (0, 0, 255) # Viền đỏ
        else:
            # Nếu tự tin cao trên 85% thì mới xét tiếp
            if class_index == 3: # Số 3 là 'nguoi_la'
                label = f"Nguoi la ({confidence*100:.1f}%)"
                color = (0, 0, 255) # Viền đỏ
            else:
                label = f"{CATEGORIES[class_index]} ({confidence*100:.1f}%)"
                color = (0, 255, 0) # Viền xanh cho nhân viên
        
        # Vẽ khung vuông và viết tên
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # Hiện cửa sổ camera
    cv2.imshow("He Thong Cham Cong AI - STU", frame)

    # Chờ 1ms, nếu bấm 'q' thì thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp sau khi tắt
cap.release()
cv2.destroyAllWindows()