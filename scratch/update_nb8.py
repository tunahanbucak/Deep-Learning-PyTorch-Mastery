import json

notebook_path = "08-LinearVsNonLinear.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Helper function to create markdown cell
def md_cell(source_text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.split("\n")]
    }

# Helper function to create code cell
def code_cell(source_text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_text.split("\n")]
    }

new_cells = []

# --- CELL 0: TITLE & INTRO ---
new_cells.append(md_cell("""# 🎯 PyTorch ile İkili Sınıflandırma (Binary Classification) ve Doğrusal (Linear) Karar Sınırları

Bu notebook'ta **PyTorch** kullanarak ikili sınıflandırma (Binary Classification) problemlerini baştan sona ele alıyoruz.

### 📌 Öğrenilecek Ana Konular:
1. **Veri Seti Hazırlığı:** Tabular e-posta verisini okuma, Scikit-Learn ile Train/Test ayırımı ve PyTorch Tenzörlerine (`float32`, `unsqueeze(1)`) dönüştürme.
2. **Sınıflandırma Mimarısı:** `nn.Module` ve `nn.Linear` katmanları ile çok katmanlı doğrusal model inşası.
3. **Tahmin Dönüşüm Zinciri (Prediction Pipeline):**
   $$\\text{Raw Logits } (-\\infty, +\\infty) \\xrightarrow{\\text{torch.sigmoid()}} \\text{Olasılıklar } [0, 1] \\xrightarrow{\\text{torch.round()}} \\text{Sınıf Etiketleri } \\{0, 1\\}$$
4. **Kayıp Fonksiyonu Farkı:** `nn.BCELoss` vs `nn.BCEWithLogitsLoss` (Sayısal kararlılık / Log-Sum-Exp hilesi).
5. **Eğitim Döngüsü & Başarı Metriği:** Manuel `Accuracy` (Doğruluk) metriği yazımı ve 100 Epochluk eğitim.
6. **Matematiksel Karar Sınırı (Decision Boundary):** Doğrusal katmanların aktivasyon fonksiyonu (ReLU) olmadan neden sadece düz bir çizgi çizebildiğinin matematiksel ispatı."""))

# --- CELL 1: IMPORTS ---
new_cells.append(md_cell("""## 1. Gerekli Kütüphanelerin Yüklenmesi"""))
new_cells.append(code_cell("""import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch versiyonu ve cihaz kontrolü
print(f"PyTorch Versiyonu: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Kullanılabilir Donanım İvmelendirici: {device}")"""))

# --- CELL 2: DATA LOADING & EDA ---
new_cells.append(md_cell("""## 2. Veri Setinin Yüklenmesi ve Keşifsel Veri Analizi (EDA)

Bu senaryoda e-postaların kişisel (Personal = 0) mi yoksa iş e-postası (Work = 1) mı olduğunu 2 öznitelik (feature) kullanarak tahmin etmeye çalışacağız:
- **`subject_formality_score`**: Konu başlığının resmiyet derecesi.
- **`sender_relationship_score`**: Gönderici ile olan samimiyet/ilişki skoru.
- **`email_type`**: Hedef değişken (Target) $\\rightarrow$ `0 = Kişisel (Personal)`, `1 = İş (Work)`."""))

new_cells.append(code_cell("""# CSV veri setini Pandas DataFrame olarak oku
df = pd.read_csv("08-email_classification_svm.csv")

# İlk 5 satırı görüntüle
df.head()"""))

new_cells.append(code_cell("""# Veri seti hakkında genel bilgi (Veri tipleri ve eksik değer kontrolü)
df.info()"""))

new_cells.append(code_cell("""# Hedef değişkendeki benzersiz sınıfları kontrol et (0 veya 1 olmalı)
print("Benzersiz Sınıflar (email_type):", df["email_type"].unique())"""))

new_cells.append(code_cell("""# İstatistiksel özet (Ortalama, Standart Sapma, Min/Max)
df.describe()"""))

new_cells.append(md_cell("""### 📊 Verinin Dağılımını Görselleştirme (Scatter Plot)
İki özniteliğin birbiriyle ilişkisini ve sınıfların uzaydaki dağılımını grafik üzerinde inceliyoruz."""))

new_cells.append(code_cell("""plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['subject_formality_score'],
    y=df['sender_relationship_score'],
    hue=df['email_type'],
    palette="Set1",
    s=60
)
plt.title("E-posta Sınıflandırma Veri Dağılımı (Kişisel vs İş)")
plt.xlabel("Konu Resmiyet Skoru (subject_formality_score)")
plt.ylabel("Gönderici İlişki Skoru (sender_relationship_score)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()"""))

# --- CELL 3: TRAIN TEST SPLIT & TENSORS ---
new_cells.append(md_cell("""## 3. Özellik (X) ve Hedef (y) Ayrımı & PyTorch Tenzörlerine Dönüştürme

> [!IMPORTANT]
> **PyTorch Tenzör Gereksinimleri:**
> 1. **Veri Tipi (`dtype`):** Katman ağırlıkları varsayılan olarak `float32` olduğu için veriler `torch.float32` yapılmalıdır.
> 2. **Boyut Uyumu (`unsqueeze(1)`):** Skalar etiketlerin (1D array) boyutu `(N,)` halinden `nn.BCEWithLogitsLoss` beklentisi olan 2D `(N, 1)` haline getirilmelidir."""))

new_cells.append(code_cell("""# 1. Pandas DataFrame'den NumPy matrislerine geçiş
X = df[["subject_formality_score", "sender_relationship_score"]].values  # Girdiler (2 Özellik)
y = df["email_type"].values                                              # Hedef Etiket (0 veya 1)

print(f"X boyutu: {X.shape} | y boyutu: {y.shape}")"""))

new_cells.append(code_cell("""from sklearn.model_selection import train_test_split

# %80 Eğitim (Train), %20 Test olacak şekilde veriyi bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim Kümesi Büyüklüğü (Train): {len(X_train)} örnek")
print(f"Test Kümesi Büyüklüğü (Test): {len(X_test)} örnek")"""))

new_cells.append(code_cell("""# NumPy dizilerini PyTorch float32 Tenzörlerine dönüştürme
X_train = torch.tensor(X_train, dtype=torch.float32)   # Shape: (800, 2)
X_test  = torch.tensor(X_test,  dtype=torch.float32)   # Shape: (200, 2)

# Hedef etiketleri 2D sütun vektörüne çevirme (unsqueeze(1))
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Shape: (800, 1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)  # Shape: (200, 1)

print("Tenzör Boyutları:")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")"""))

# --- CELL 4: MODEL DEFINITION ---
new_cells.append(md_cell("""## 4. PyTorch Sınıflandırma Modelinin Tanımlanması (`nn.Module`)

Bu modelde 2 doğrusal katman (`nn.Linear`) kullanıyoruz:
- **`layer_1`**: 2 girdi özelliğini alır (`in_features=2`), 5 gizli nöron üretir (`out_features=5`).
- **`layer_2`**: 5 gizli nöronu alır (`in_features=5`), 1 ham lojit (logit) çıktısı verir (`out_features=1`).

> [!WARNING]
> Bu aşamada katmanlar arasında **hiçbir Aktivasyon Fonksiyonu (ReLU vb.) KULLANMIYORUZ**.
> İki doğrusal katmanın arka arkaya dizilmesi matematiksel olarak tek bir doğrusal katmana eşdeğerdir:
> $$W_2 \\cdot (W_1 \\cdot X + b_1) + b_2 = (W_2 W_1) X + (W_2 b_1 + b_2) = W_{\\text{eff}} X + b_{\\text{eff}}$$
> Bu model sadece **Doğrusal (Linear) Karar Sınırı (Çizgi)** çizebilir."""))

new_cells.append(code_cell("""from torch import nn

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Katman: 2 girdi -> 5 gizli nöron
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        # 2. Katman: 5 gizli nöron -> 1 çıktı (Lojit)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # İleri yayılım: Veri sırasıyla layer_1 ve layer_2'den geçer
        return self.layer_2(self.layer_1(x))

# Model nesnesini oluştur
model_0 = ClassificationModel()
print("Model Mimarisi:")
print(model_0)"""))

# --- CELL 5: LOSS & OPTIMIZER & ACCURACY ---
new_cells.append(md_cell("""## 5. Kayıp Fonksiyonu (Loss Function), Optimizasyon ve Başarı Metriği

### 💡 Neden `nn.BCEWithLogitsLoss()` Seçtik?
1. **`nn.BCELoss()`**: Girdi olarak önceden `torch.sigmoid()` uygulanmış **olasılıkları** $[0, 1]$ bekler.
2. **`nn.BCEWithLogitsLoss()`**: Girdi olarak modelin ürettiği **ham lojitleri (raw logits)** $(-\\infty, +\\infty)$ bekler.
   * İçerisinde Sigmoid ve Binary Cross Entropy işlemlerini **Log-Sum-Exp hilesi** ile birleştirdiği için sayısal olarak **çok daha kararlıdır (numerically stable)** ve patlamaları önler."""))

new_cells.append(code_cell("""# Kayıp Fonksiyonu: Binary Cross Entropy with Logits
loss_fn = nn.BCEWithLogitsLoss()

# Optimizasyon Algoritması: Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Manuel Doğruluk (Accuracy) Metriği Fonksiyonu
def calculate_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    \"\"\"Gerçek etiketler ile tahmin edilen etiketler arasındaki doğruluk yüzdesini hesaplar.\"\"\"
    correct = torch.eq(y_true, y_pred).sum().item()  # Eşit olan elemanların toplamı
    acc = (correct / len(y_pred)) * 100
    return acc"""))

# --- CELL 6: LOGITS TO PROBS TO PREDS DEMO ---
new_cells.append(md_cell("""## 6. Sınıflandırmada Tahmin Mantığı: Logits $\\rightarrow$ Sigmoid $\\rightarrow$ Pred Labels

Model eğitilmeden önce ürettiği ham çıktıları adım adım inceliyoruz:

```
    [ Model ] ──> Raw Logits (-inf, +inf) 
                      │
              torch.sigmoid()
                      ▼
               Pred Probs [0.0, 1.0] 
                      │
               torch.round()  ( Threshold = 0.5 )
                      ▼
               Pred Labels {0, 1}
```"""))

new_cells.append(code_cell("""# Eğitilmemiş modelden ilk 5 test örneği için tahmin alıyoruz
y_logits = model_0(X_test)[:5]
print("1. Ham Lojitler (Raw Logits):\\n", y_logits)"""))

new_cells.append(code_cell("""# Lojitleri Sigmoid fonksiyonundan geçirerek 0 ile 1 arasında olasılıklara dönüştürüyoruz
y_pred_probs = torch.sigmoid(y_logits)
print("2. Olasılık Değerleri (Prediction Probabilities):\\n", y_pred_probs)"""))

new_cells.append(code_cell("""# Olasılıkları 0.5 eşik değerine (threshold) göre 0 veya 1 sınıf etiketlerine yuvarlıyoruz
# >= 0.5 ise 1 (İş e-postası), < 0.5 ise 0 (Kişisel e-posta)
y_preds = torch.round(y_pred_probs)
print("3. Tahmin Edilen Sınıf Etiketleri (Prediction Labels):\\n", y_preds)
print("\\nGerçek Etiketler (True Labels):\\n", y_test[:5])"""))

new_cells.append(code_cell("""# Eğitimsiz modelin ilk 5 örnek üzerindeki doğruluğu
print(f"Eğitimsiz İlk 5 Doğruluk Skoru: %{calculate_accuracy(y_test[:5], y_preds):.2f}")"""))

# --- CELL 7: TRAINING LOOP ---
new_cells.append(md_cell("""## 7. Model Eğitim Döngüsü (Training Loop - 100 Epoch)

100 epoch boyunca standart 5 adımlı PyTorch eğitim döngüsünü çalıştırıyoruz."""))

new_cells.append(code_cell("""torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    # --- 1. EĞİTİM MODU ---
    model_0.train()

    # İleri Yayılım (Raw Logits üretir)
    y_logits = model_0(X_train)
    
    # Metrik hesabı için Lojit -> Olasılık -> Etiket dönüşümü
    y_pred = torch.round(torch.sigmoid(y_logits))
  
    # Kayıp (Loss) Hesaplama: Raw Logits ve Gerçek Etiketler
    loss = loss_fn(y_logits, y_train)
    acc = calculate_accuracy(y_true=y_train, y_pred=y_pred)

    # Gradyanları Sıfırla
    optimizer.zero_grad()

    # Geri Yayılım (Backpropagation)
    loss.backward()

    # Parametre Güncelleme (Step)
    optimizer.step()

    # --- 2. DEĞERLENDİRME MODU (TEST) ---
    model_0.eval()
    with torch.inference_mode():
        # İleri Yayılım
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # Test Kaybı ve Doğruluğu
        test_loss = loss_fn(test_logits, y_test)
        test_acc = calculate_accuracy(y_true=y_test, y_pred=test_pred)

    # Her 5 epoch'ta bir durumu ekrana yazdır
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"Epoch: {epoch:3d} | Loss: {loss:.5f} | Acc: %{acc:.2f} | Test Loss: {test_loss:.5f} | Test Acc: %{test_acc:.2f}")"""))

# --- CELL 8: DECISION BOUNDARY VISUALIZATION ---
new_cells.append(md_cell("""## 8. Karar Sınırını (Decision Boundary) Görselleştirme

Bu aşamada eğitilen modelin uzaydaki iki sınıfı birbirinden nasıl bir çizgi ile ayırdığını matematiksel doğrusal denklem üzerinden çizdiriyoruz.

### 📐 Matematiksel İspat:
Modelimiz iki katmandan oluşur:
$$W_1 \\in \\mathbb{R}^{5 \\times 2}, \\quad b_1 \\in \\mathbb{R}^{5}$$
$$W_2 \\in \\mathbb{R}^{1 \\times 5}, \\quad b_2 \\in \\mathbb{R}^{1}$$

Aktivasyon fonksiyonu (ReLU) olmadığı için efektif ağırlık ve bias:
$$W_{\\text{eff}} = W_2 \\cdot W_1 = [w_1, w_2]$$
$$b_{\\text{eff}} = W_2 \\cdot b_1 + b_2$$

Karar sınırı denklemimiz ($w_1 x_1 + w_2 x_2 + b_{\\text{eff}} = 0$):
$$x_2 = \\frac{-(w_1 x_1 + b_{\\text{eff}})}{w_2}$$"""))

new_cells.append(code_cell("""import numpy as np

def plot_linear_decision_boundary(model, X, y):
    \"\"\"Eğitilmiş doğrusal sınıflandırma modelinin karar sınırını (çizgisini) çizer.\"\"\"
    # Model katman ağırlıklarını ve biaslarını al
    W1 = model.layer_1.weight.detach().numpy()       # Shape: (5, 2)
    b1 = model.layer_1.bias.detach().numpy()         # Shape: (5,)
    W2 = model.layer_2.weight.detach().numpy()       # Shape: (1, 5)
    b2 = model.layer_2.bias.detach().numpy()[0]      # Shape: (1,)

    # Efektif Doğrusal Ağırlık ve Bias Hesaplama
    W_eff = W2 @ W1   # Shape: (1, 2)
    w1, w2 = W_eff[0] # x1 (subject_formality) ve x2 (sender_relationship) ağırlıkları

    b_eff = (W2 @ b1)[0] + b2  # Efektif Bias

    # X1 ekseni için veri aralığı
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)

    # Doğru denkleminden X2 hesaplama: w1*x1 + w2*x2 + b = 0 -> x2 = -(w1*x1 + b) / w2
    ys = -(w1 * xs + b_eff) / w2

    # Veri noktalarını çizdir
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdYlBu, s=40, edgecolors="k", alpha=0.8)
    # Karar Sınırı Çizgisi
    plt.plot(xs, ys, "k-", linewidth=3, label="Karar Sınırı (Decision Boundary)")
    plt.xlabel("Subject Formality Score")
    plt.ylabel("Sender Relationship Score")
    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.legend()"""))

new_cells.append(code_cell("""plt.figure(figsize=(14, 6))

# Train Karar Sınırı
plt.subplot(1, 2, 1)
plt.title("Eğitim Kümesi (Train Set) Karar Sınırı")
plot_linear_decision_boundary(model_0, X_train.numpy(), y_train.numpy())

# Test Karar Sınırı
plt.subplot(1, 2, 2)
plt.title("Test Kümesi (Test Set) Karar Sınırı")
plot_linear_decision_boundary(model_0, X_test.numpy(), y_test.numpy())

plt.tight_layout()
plt.show()"""))

# --- CELL 9: SUMMARY TABLE ---
new_cells.append(md_cell("""## 📝 Modül Özeti ve Önemli Çıkarımlar

| Kavram / Terim | Açıklama / Kural | PyTorch Kodu / Değeri |
| :--- | :--- | :--- |
| **Problem Tipi** | İkili Sınıflandırma (Binary Classification) | 0 (Personal) vs 1 (Work) |
| **Çıktı Boyutu** | Son katmanın nöron sayısı | `out_features = 1` |
| **Logits** | Modelin ham (raw) kısıtlanmamış çıktısı | `y_logits = model(X)` $(-\\infty, +\\infty)$ |
| **Olasılık (Probabilities)** | Lojitlerin 0 ile 1 arasına sıkıştırılması | `y_pred_probs = torch.sigmoid(y_logits)` |
| **Sınıf Etiketi (Labels)** | 0.5 eşik değerine göre yuvarlama | `y_preds = torch.round(y_pred_probs)` |
| **Kayıp Fonksiyonu** | Sayısal olarak kararlı Binary Cross Entropy | `nn.BCEWithLogitsLoss()` |
| **Aktivasyonsuz Ağ** | Aktivasyon fonksiyonu olmadan 100 katman da yapılsa model doğrusal (çizgi) kalır. | `Linear(2,5) -> Linear(5,1)` |

---
*Proje Rehberi: Atıl Samancıoğlu Derin Öğrenme Eğitimi Modül 08*"""))

nb["cells"] = new_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("08-LinearVsNonLinear.ipynb successfully updated with rich Turkish documentation and code comments!")
