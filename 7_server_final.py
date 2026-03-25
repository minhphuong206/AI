from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import datetime
import csv
import os
import shutil
import base64
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

app = Flask(__name__)

# --- CẤU HÌNH HỆ THỐNG ---
LOG_FILE = "lich_su_cham_cong.csv"
DATASET_DIR = "dataset"
MODEL_PATH = "model_chamcong.h5"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tự động quét các thư mục để lấy danh sách nhân viên (Người lạ luôn nằm cuối)
def get_dynamic_categories():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != 'nguoi_la']
    categories.append('nguoi_la') # Bắt buộc nhóm người lạ nằm cuối
    return categories

CATEGORIES = get_dynamic_categories()

# Load Model nếu có
try:
    model = load_model(MODEL_PATH)
    print(" Đã tải thành công bộ não AI.")
except:
    model = None
    print(" CẢNH BÁO: Chưa có bộ não AI. Hãy vào /admin để Huấn luyện!")

# --- 1. CHỨC NĂNG CHẤM CÔNG ---
last_logged = {}
latest_log = {"name": "", "time": "", "status": "waiting"}

def generate_frames():
    global latest_log, last_logged, model, CATEGORIES
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.2, 5, minSize=(120, 120))

        for (x, y, w, h) in faces:
            if model is None:
                cv2.putText(frame, "CHUA DUOC HUAN LUYEN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                continue

            face_roi = gray_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (64, 64))
            reshaped_face = np.reshape(resized_face.astype('float32') / 255.0, (1, 64, 64, 1))

            predictions = model.predict(reshaped_face, verbose=0)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions)

            if confidence >= 0.90:
                if CATEGORIES[class_index] == 'nguoi_la':
                    label, color = f"Nguoi la ({confidence*100:.1f}%)", (0, 0, 255)
                else:
                    name = CATEGORIES[class_index]
                    label, color = f"{name} ({confidence*100:.1f}%)", (0, 255, 0)
                    now = datetime.datetime.now()
                    
                    if name not in last_logged or (now - last_logged[name]).total_seconds() > 60:
                        time_str = now.strftime("%d/%m/%Y %H:%M:%S")
                        if not os.path.exists(LOG_FILE):
                            with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f: csv.writer(f).writerow(["Tên", "Thời Gian"])
                        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                            csv.writer(f).writerow([name, time_str])
                        last_logged[name] = now
                        latest_log = {"name": name, "time": time_str, "status": "success"}
            else:
                label, color = f"Nguoi la ({confidence*100:.1f}%)", (0, 0, 255)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()


# --- 2. CHỨC NĂNG QUẢN TRỊ (ADMIN) ---

# Hàm giải mã Base64 và cắt khuôn mặt
def process_and_augment(b64_string, output_dir):
    img_data = base64.b64decode(b64_string.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    if len(faces) == 0: return False # Không thấy mặt
    
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    base_face = cv2.resize(face_roi, (64, 64))

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "base.jpg"), base_face)

    # Tự động sinh ra 30 tấm nhiễu
    for i in range(1, 30):
        angle = np.random.uniform(-10, 10) 
        M = cv2.getRotationMatrix2D((32, 32), angle, 1)
        rotated = cv2.warpAffine(base_face, M, (64, 64))
        brightened = cv2.convertScaleAbs(rotated, alpha=np.random.uniform(0.7, 1.3), beta=0)
        noise = np.random.normal(0, 7, brightened.shape)
        final_img = np.clip(brightened.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"aug_{i}.jpg"), final_img)
    return True

@app.route('/api/add_employee', methods=['POST'])
def add_employee():
    data = request.json
    name = data['name'].replace(" ", "_") # Chuyển dấu cách thành _ cho tên thư mục
    images = data['images']
    
    # Xử lý 3 góc
    for angle_name, b64_img in images.items():
        out_dir = os.path.join(DATASET_DIR, name, angle_name)
        if not process_and_augment(b64_img, out_dir):
            return jsonify({"success": False, "message": f"Không tìm thấy khuôn mặt ở ảnh {angle_name}. Hãy chụp gần và sáng hơn!"})
    
    return jsonify({"success": True})

@app.route('/api/get_employees')
def get_employees():
    cats = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d)) and d != 'nguoi_la']
    return jsonify({"employees": cats})

@app.route('/api/delete_employee', methods=['POST'])
def delete_employee():
    name = request.json['name']
    path = os.path.join(DATASET_DIR, name)
    if os.path.exists(path): shutil.rmtree(path)
    return jsonify({"success": True})

@app.route('/api/retrain', methods=['POST'])
def retrain():
    global model, CATEGORIES
    CATEGORIES = get_dynamic_categories()
    
    X, y = [], []
    for class_num, category in enumerate(CATEGORIES):
        path = os.path.join(DATASET_DIR, category)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.shape == (64, 64):
                        X.append(img)
                        y.append(class_num)
                        
    X = np.array(X).astype('float32') / 255.0
    X = X.reshape(-1, 64, 64, 1)
    y = np.array(y)

    new_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)), MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'), MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'), Dropout(0.5),
        Dense(len(CATEGORIES), activation='softmax')
    ])
    new_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(X, y, epochs=20, verbose=0) # verbose=0 để chạy ngầm không in log
    new_model.save(MODEL_PATH)
    
    model = new_model # Cập nhật não bộ trực tiếp không cần reset server
    return jsonify({"success": True})

# --- ĐỊNH TUYẾN GIAO DIỆN ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/admin')
def admin(): return render_template('admin.html')

# === BẮT ĐẦU ĐOẠN CODE BỔ SUNG ===
@app.route('/history')
def history():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            logs = list(reader)
    logs.reverse()
    return render_template('history.html', logs=logs)
# === KẾT THÚC ĐOẠN CODE BỔ SUNG ===

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_status')
def check_status():
    global latest_log
    current_log = latest_log.copy()
    if latest_log["status"] == "success": latest_log["status"] = "waiting"
    return jsonify(current_log)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)