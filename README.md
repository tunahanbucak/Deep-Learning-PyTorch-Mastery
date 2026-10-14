# 🧠 PyTorch & Derin Öğrenme (Deep Learning) Uçtan Uca Rehber

Bu depo, **Derin Öğrenme (Deep Learning)** teorisini ve **PyTorch** kütüphanesini sıfırdan ileri seviyeye kadar adım adım, detaylı açıklamalar ve uygulama kodları ile öğrenmek amacıyla hazırlanmış kapsamlı bir eğitim ve çalışma alanıdır.

Proje, hem matematiği ve arka plandaki algoritmaları (**Teorik Katman**) hem de kodlama ve mimari tasarım yeteneklerini (**Pratik Katman**) kapsayan **çift katmanlı (Dual-Layer)** bir öğrenme yaklaşımına sahiptir.

---

## 🏗️ Proje Mimarisi ve Katmanlar

```
📁 DeepLearningNotebooks
├── 📄 README.md                                  # Canlı Proje Yol Haritası ve İndeks
├── 📁 Derin Öğrenme Notlar/
│   └── 📄 0-Derin_Ogrenme_Teori_Master_Rehberi.md # 23 Teorik PDF'in Master Özeti & Matematiği
└── 📓 Jupyter Notebooks (01 - 16)               # Adım Adım PyTorch Kodlama & Uygulamalar
```

---

## 📌 İçindekiler ve Modül Durumu

### 📘 Katman 1: Teorik Master Rehber
- 📖 [**Derin Öğrenme Teori Master Rehberi**](./Derin%20Öğrenme%20Notlar/0-Derin_Ogrenme_Teori_Master_Rehberi.md)
  > Perceptron, Aktivasyon Fonksiyonları, Backpropagation (Geri Yayılım), Vanishing Gradient, Kayıp (Loss) Fonksiyonları, Optimizasyon Algoritmaları (SGD, Adam), Düzenlileştirme (Dropout, L1/L2) ve Normalizasyon (BatchNorm, LayerNorm) teorilerini içerir.

---

### 💻 Katman 2: PyTorch Pratik Notebook'ları

| Durum | Modül / Dosya | Kapsanan Konular & İçerik |
| :---: | :--- | :--- |
| ✅ | [**01-PyTorchFundamentals.ipynb**](./01-PyTorchFundamentals.ipynb) | Tenzör nedir? Skalar, Vektör, Matris, Tenzör boyutları (`ndim`, `shape`), veri tipleri (`dtype`), `item()` ve NumPy entegrasyonu. |
| ✅ | [**02-PyTorchOperations.ipynb**](./02-PyTorchOperations.ipynb) | Rastgele tenzör üretimi (`rand`), sıfır/bir tenzörleri (`zeros`, `ones`), sayı aralıkları (`arange`), Görsel formatı `(C, H, W)` ve eleman bazlı matris matematiği. |
| ✅ | [**03-PyTorchMatrixMultiplication.ipynb**](./03-PyTorchMatrixMultiplication.ipynb) | Eleman bazlı çarpım (`*`) vs Matris Çarpımı (`@` / `matmul`), iç boyut uyumu kuralı, Transpoz alımı (`.T`) ve Doğrusal Model Denklemi ($Y = XW^T + b$). |
| ✅ | [**04-PyTorchAggregationManipulation.ipynb**](./04-PyTorchAggregationManipulation.ipynb) | Min/Max/Sum/Mean istatistikleri (float32 kuralı), `argmax()`, `reshape`, `view` (bellek paylaşımı), `squeeze`, `unsqueeze(1)` ve `permute` ile kanal ekseni sıralama. |
| ✅ | [**05-PyTorchDeviceSettings.ipynb**](./05-PyTorchDeviceSettings.ipynb) | Donanım ivmelendirme (CUDA / Apple Silicon MPS), `tensor.to(device)`, cihaz uyuşmazlığı hataları ve MPS/CUDA'dan NumPy'a geçerken `.cpu().numpy()` kuralı. |
| ✅ | [**06-PyTorchTrainingSteps.ipynb**](./06-PyTorchTrainingSteps.ipynb) | Custom `nn.Module` Doğrusal Regresyon modeli, `requires_grad=True`, `inference_mode()`, `MSELoss()`, `SGD()` ve PyTorch'un 5 Adımlı Standart Eğitim Döngüsü. |
| ✅ | [**07-PyTorchTrainingStructural.ipynb**](./07-PyTorchTrainingStructural.ipynb) | Dahili `nn.Linear(in_features, out_features)` kullanımı, `dtype=torch.float32` zorunluluğu, 2D `(batch_size, in_features)` boyut hazırlığı (`.unsqueeze(1)`), kayıp takibi ve görselleştirme. |
| ✅ | [**08-LinearVsNonLinear.ipynb**](./08-LinearVsNonLinear.ipynb) | İkili Sınıflandırma (Binary Classification), Logits -> Sigmoid -> Pred Labels dönüşüm zinciri, `nn.BCEWithLogitsLoss`, manuel Accuracy metriği ve Doğrusal Karar Sınırı (Decision Boundary) çizimi. |
| ✅ | [**08-HomeworkSeismic.ipynb**](./08-HomeworkSeismic.ipynb) | **Ödev & Lab:** Sismik Aktivite Sınıflandırması. Doğrusal modelin çöküşü (Test Acc: %47.50) vs `nn.ReLU()` Non-Linear modelin zaferi (Test Acc: %100.00) ve Karar Sınırı kıyaslaması. |
| ✅ | [**09-MultiClassClassification.ipynb**](./09-MultiClassClassification.ipynb) | Çoklu Sınıflandırma (Multi-Class), Iris Veri Seti, Logits -> Softmax -> Argmax zinciri, `nn.CrossEntropyLoss`, `torchmetrics` Confusion Matrix ve Model Kaydetme/Yükleme (`.pth`). |
| ✅ | [**HomeMultiClassClassification.ipynb**](./HomeMultiClassClassification.ipynb) | **Ödev & Lab:** Palmer Penguins Çoklu Sınıflandırma. Eksik veri (`NaN`) yönetimi, `LabelEncoder`, `StandardScaler` vaka analizi (%43'ten %100'e başarı sıçraması), Confusion Matrix ve Canlı Tahmin (Inference). |
| 🔄 | **10 - PyTorch DataLoader & Dataset Yapısı** | *(İşlenecek)* Özel Veri Setleri (Custom Datasets), `Dataset` sınıfı, `DataLoader`, Batchleme ve Shuffle mekanizması. |
| 🔄 | **11 - Model Değerlendirme & Metrikler** | *(İşlenecek)* Accuracy, Precision, Recall, F1-Score, Confusion Matrix ve TensorBoard entegrasyonu. |
| 🔄 | **12 - Transfer Learning (Transfer Öğrenme)** | *(İşlenecek)* Önceden eğitilmiş modeller (ResNet, EfficientNet vb.), Feature Extraction ve Fine-Tuning. |
| 🔄 | **13-16 - İleri Seviye PyTorch & Projeler** | *(İşlenecek)* Gerçek hayat uçtan uca derin öğrenme projeleri ve yayınlama (Deployment). |

---

## ⚡ PyTorch 5-Adımlı Eğitim Döngüsü (Hızlı Referans)

PyTorch'taki her model eğitimi aşağıdaki standart 5 adımı takip eder:

```python
# 1. Modeli Eğitim Moduna Al
model.train()

# 2. İleri Yayılım (Forward Pass)
y_pred = model(X_train)

# 3. Kayıp Hesaplama (Loss Calculation)
loss = loss_fn(y_pred, y_train)

# 4. Gradyanları Sıfırla (Zero Gradients)
optimizer.zero_grad()

# 5. Geri Yayılım & Parametre Güncelleme (Backward Pass & Step)
loss.backward()
optimizer.step()
```

---

## 🛠️ Kurulum ve Çalıştırma Ortamı

1. **Gerekli Kütüphaneler:**
   ```bash
   pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
2. **Apple Silicon (M1/M2/M3/M4) veya CUDA Kontrolü:**
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
   print(f"Aktif Cihaz: {device}")
   ```

---

## 🎯 Hedef ve Yol Haritası
Bu depo, Atıl Samancıoğlu'nun Derin Öğrenme eğitimi ve sektör standartlarındaki PyTorch pratikleri doğrultusunda hazırlanmaktadır. Amacımız, **Uçtan Uca Yapay Zeka Uygulamaları & Full-Stack AI Engineer** yetkinliğine ulaşmaktır.
