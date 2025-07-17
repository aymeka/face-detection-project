# YOLO Face Detection Project

**Gerçek zamanlı yüz algılama sistemi (YOLO tabanlı)**\
Flask tabanlı web uygulamasıyla kameradan canlı yüz algılama.

---

## 🗂️ Proje Yapısı

```
.
├── .idea/                 # IDE ayarları (isteğe bağlı)
├── data/                  # Veri setleri (eğitim/test için)
├── flask_app/             # Flask uygulaması (app.py, templates, static)
├── models/                # Eğitilmiş YOLO modelleri (best.pt burada)
├── results/               # Sonuç çıktıları
├── scripts/               # Eğitim, değerlendirme, vs. betikleri
├── README                 # Bu dosya
└── requirements           # Gereksinimler
```

---

## 🚀 Kurulum

📦 Gerekli paketleri yükleyin:

```bash
pip install -r requirements
```

Eğitim ya da kendi modelinizi hazırladıktan sonra, model dosyanızı `models/best.pt` olarak yerleştirin.

---

## 🧪 Eğitim

Eğitim ve değerlendirme betikleri `scripts/` klasöründedir.\
Örneğin eğitim için:

```bash
python scripts/train_yolo.py
```

---

## 🌐 Çalıştırma

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

## 📄 Notlar

✅ Şu anda yalnızca **YOLO tabanlı model** desteklenmektedir.\
✅ Modelin eğitimli hali (`best.pt`) `models/` klasöründe bulunmalıdır.\
✅ `results/` klasörü, eğitim/evaluasyon çıktıları için ayrılmıştır.

---

## 📃 Lisans

Bu proje açık kaynaklıdır.\
Eğitim, araştırma ve kişisel kullanım içindir.

