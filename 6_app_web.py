from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime
import csv
import os

app = Flask(__name__)

print(" Đang khởi động Server và AI...")
model = load_model('model_chamcong.h5')
CATEGORIES = ["Nhan vien 1", "Nhan vien 2", "Phuong (NV3)", "Nguoi la"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tên file cơ sở dữ liệu (sẽ tự động tạo nếu chưa có)
LOG_FILE = "lich_su_cham_cong.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Tên Nhân Viên", "Thời Gian"]) # Tạo tiêu đề cột

# Biến lưu thời gian chấm công cuối cùng của từng người (Chống spam)
last_logged = {}
# Biến báo cáo lên giao diện web
latest_log = {"name": "", "time": "", "status": "waiting"}

def generate_frames():
    global latest_log, last_logged
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y+h, x:x+w]
            
            resized_face = cv2.resize(face_roi, (64, 64))
            normalized_face = resized_face.astype('float32') / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

            predictions = model.predict(reshaped_face, verbose=0)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            now = datetime.datetime.now()

            # NGƯỠNG TỰ TIN 90%
            if confidence >= 0.90:
                if class_index == 3: # Người lạ
                    label = f"Nguoi la ({confidence*100:.1f}%)"
                    color = (0, 0, 255)
                else: # Nhân viên hợp lệ
                    name = CATEGORIES[class_index]
                    label = f"{name} ({confidence*100:.1f}%)"
                    color = (0, 255, 0)
                    
                    # LOGIC GHI LỊCH SỬ CHẤM CÔNG VÀ CHỐNG SPAM (60 giây)
                    if name not in last_logged or (now - last_logged[name]).total_seconds() > 60:
                        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
                        
                        # Mở file CSV và ghi thêm 1 dòng
                        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([name, time_str])
                        
                        last_logged[name] = now # Cập nhật lại thời gian vừa quét
                        latest_log = {"name": name, "time": time_str, "status": "success"}
            else:
                label = f"Khong xac dinh ({confidence*100:.1f}%)"
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ĐỊNH TUYẾN WEB ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_status')
def check_status():
    global latest_log
    # Copy dữ liệu để gửi đi, sau đó reset thành waiting để không báo liên tục
    current_log = latest_log.copy()
    if latest_log["status"] == "success":
        latest_log["status"] = "waiting"
    return jsonify(current_log)

# TAB MỚI: Trang xem lịch sử
@app.route('/history')
def history():
    logs = []
    # Đọc dữ liệu từ file Excel CSV
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None) # Bỏ qua dòng tiêu đề
            logs = list(reader)
    
    logs.reverse() # Đảo ngược để người mới chấm công hiện lên đầu danh sách
    return render_template('history.html', logs=logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)