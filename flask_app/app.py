from flask import Flask, render_template, Response
import cv2
import os

from ultralytics import YOLO

app = Flask(__name__)

# En iyi modeli yükle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
model = YOLO(MODEL_PATH)

# Kamera (başta None)
camera = None


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Modeli çalıştır
        results = model(frame)

        # Sonuçları çiz
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = [int(i) for i in box]
                label = f"Face: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # JPEG olarak encode et
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    while True:
        try:
            cam_id = int(input("Kullanmak istediğiniz kamera ID’sini girin (ör. 0, 1, 2…): "))
            camera = cv2.VideoCapture(cam_id)
            if not camera.isOpened():
                print(f"Kamera {cam_id} açılamadı. Başka bir ID deneyin.")
                continue
            print(f"Kamera {cam_id} başarıyla açıldı.")
            break
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")

    app.run(host='0.0.0.0', port=5000, debug=True)
