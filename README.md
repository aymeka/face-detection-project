# YOLO Face Detection Project

github linki: https://github.com/aymeka/face-detection-project

Ayça Melisa Karadede

Yapay Zeka

**Gerçek zamanlı yüz algılama sistemi (YOLO tabanlı)**\
Flask tabanlı web uygulamasıyla kameradan canlı yüz algılama.

---

## Proje Yapısı

```
.
├── .idea/                 # IDE ayarları (isteğe bağlı)
├── data/                  # Veri setleri (eğitim/test için)
├── flask_app/             # Flask uygulaması (app.py, templates, static)
├── models/                # Eğitilmiş YOLO modeli (best.pt burada)
├── results/               # Sonuç çıktıları
├── scripts/               # Eğitim, değerlendirme, vs. betikleri
├── README                 
└── requirements           # Gereksinimler
```

---

## Kurulum

📦 Gerekli paketleri yükleyin:

```bash
pip install -r requirements
```

Eğitim ya da kendi modelinizi hazırladıktan sonra, model dosyanızı `models/best.pt` olarak yerleştirin.

---

## Eğitim

Eğitim ve değerlendirme betikleri `scripts/` klasöründedir.\
Örneğin eğitim için:

```bash
python scripts/train_yolo.py
```

---

## Çalıştırma

Web uygulamasını başlatmak için:

```bash
cd flask_app
python app.py
```

Tarayıcıda ziyaret edin:

```
http://localhost:5000
```

Kamera akışında algılanan yüzleri göreceksiniz.

---

## Notlar

Şu anda yalnızca **YOLO tabanlı model** desteklenmektedir.\
Modelin eğitimli hali (`best.pt`) `models/` klasöründe bulunmalıdır.\
`results/` klasörü, eğitim/evaluasyon çıktıları için ayrılmıştır.

## 📁 Proje Adımları & Commit Geçmişi

### 📦 Veri Seti
- 2025-07-18: Add initial dataset (train, val, test)

### 🧠 Model Eğitimi
- 2025-07-18: Add YOLOv8 training script
- 2025-07-19: Trained YOLOv8 and added evaluation metrics

### 📊 Değerlendirme
- 2025-07-20: Add confusion matrix and F1 score visualization





