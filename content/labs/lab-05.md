## ข้อมูลทั่วไปของใบงาน

| รายการ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 05: K-Means, PCA และการเลือกอัลกอริทึม |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 6 |
| งานตาม มคอ.3 | แบบฝึกหัด 4: เลือกอัลกอริทึมให้เหมาะสม |
| CLO/LLO | CLO2 (LLO2-2 เลือกใช้ clustering, dimensionality reduction, anomaly detection ให้เหมาะกับปัญหา; LLO2-3 พัฒนาและประเมินผลด้วย scikit-learn) |
| เครื่องมือ | Google Colab, Python 3, NumPy, pandas, matplotlib, scikit-learn |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ + ตารางคำนวณมือข้อ 1 + คำตอบคำถามท้ายข้อ |

### วัตถุประสงค์

1. คำนวณ K-Means ด้วยมือ 2 รอบ แล้วตรวจคำตอบด้วย NumPy และ scikit-learn ได้
2. เลือกจำนวนกลุ่มด้วย Elbow และ Silhouette และอธิบายผลของ feature scaling ต่อการจัดกลุ่มได้
3. เปรียบเทียบ K-Means กับ DBSCAN บนข้อมูลที่ไม่ใช่ทรงกลม และปรับค่า `eps` ได้
4. ใช้ PCA ลดมิติ เลือกจำนวน component จาก explained variance และสร้างภาพกลับ (reconstruction) ได้
5. ตรวจจับความผิดปกติด้วย Gaussian Mixture และ Isolation Forest และวิเคราะห์ผลของ threshold ได้
6. วิเคราะห์โจทย์จริงแล้วเลือกเทคนิค unsupervised ที่เหมาะสมพร้อมให้เหตุผลได้

## ข้อที่ 0: เตรียมสภาพแวดล้อม

รันเซลล์นี้ก่อนทุกครั้ง ทุกชุดข้อมูลในใบงานนี้สร้างหรือโหลดจาก scikit-learn จึงไม่ต้องอัปโหลดไฟล์

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_moons, load_digits

np.set_printoptions(suppress=True)
print("พร้อมแล้ว")
```

## ข้อที่ 1: K-Means ด้วยมือ

### โจทย์

ข้อมูล 8 จุดต่อไปนี้ ให้แบ่งเป็น **K = 3** กลุ่ม โดยใช้ระยะ Euclidean และกำหนด centroid เริ่มต้นเป็น **C1 = A1, C2 = A4, C3 = A7**

| จุด | A1 | A2 | A3 | A4 | A5 | A6 | A7 | A8 |
|---|---|---|---|---|---|---|---|---|
| (x, y) | (2, 10) | (2, 5) | (8, 4) | (5, 8) | (7, 5) | (6, 4) | (1, 2) | (4, 9) |

**1.1 คำนวณด้วยมือ (ทำในกระดาษหรือเซลล์ markdown)** — เติมตารางระยะทางของรอบที่ 1 และรอบที่ 2 ระบุกลุ่มของแต่ละจุด และคำนวณ centroid ใหม่หลังแต่ละรอบ (ทศนิยม 2 ตำแหน่ง)

| จุด | d(C1) | d(C2) | d(C3) | กลุ่ม |
|---|---|---|---|---|
| A1 (2, 10) | | | | |
| A2 (2, 5) | | | | |
| ... | | | | |
| A8 (4, 9) | | | | |

**1.2 เขียนโค้ดตรวจคำตอบ** — เติมฟังก์ชัน `assign_clusters()` และ `update_centroids()` ให้ถูกต้อง

**1.3 ตรวจกับ scikit-learn** — รัน `KMeans` ด้วย centroid เริ่มต้นเดียวกันจนลู่เข้า

### โค้ดเริ่มต้น

```python
X1 = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]], dtype=float)
names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
C0 = X1[[0, 3, 6]].copy()      # centroid เริ่มต้น = A1, A4, A7

def assign_clusters(X, C):
    """คืน (distances, labels) โดย distances มี shape (จำนวนจุด, k)"""
    # TODO: คำนวณระยะ Euclidean จากทุกจุดไปยังทุก centroid
    #       คำใบ้: X[:, None, :] - C[None, :, :] ได้ array shape (n, k, 2) แล้วยกกำลังสอง รวมตามแกน 2 และถอดราก
    distances = np.zeros((len(X), len(C)))
    labels = distances.argmin(axis=1)
    return distances, labels

def update_centroids(X, labels, k):
    """คืน centroid ใหม่ shape (k, 2) = ค่าเฉลี่ยของจุดในแต่ละกลุ่ม"""
    # TODO: ใช้ X[labels == j].mean(axis=0) สำหรับ j = 0, 1, ..., k-1
    return np.zeros((k, X.shape[1]))

C = C0.copy()
for it in range(1, 3):
    D, labels = assign_clusters(X1, C)
    print(f"--- รอบที่ {it} ---")
    print(pd.DataFrame(D.round(2), index=names, columns=["d(C1)", "d(C2)", "d(C3)"]).assign(cluster=labels + 1))
    C = update_centroids(X1, labels, 3)
    print("centroid ใหม่:", C.round(3).tolist())
```

```python
# 1.3 ตรวจกับ scikit-learn: init=C0 และ n_init=1 เพื่อเริ่มจากจุดเดียวกับการคำนวณมือ
km1 = KMeans(n_clusters=3, init=C0, n_init=1).fit(X1)
print("sklearn labels   :", km1.labels_ + 1)
print("sklearn centroids:", km1.cluster_centers_.round(3).tolist())
print("inertia:", round(km1.inertia_, 3), "| iterations:", km1.n_iter_)
```

### ผลลัพธ์ที่คาดหวัง

<details><summary>เปิดดูหลังคำนวณมือเสร็จแล้ว</summary>

```output
--- รอบที่ 1 ---
    d(C1)  d(C2)  d(C3)  cluster
A1   0.00   3.61   8.06        1
A2   5.00   4.24   3.16        3
A3   8.49   5.00   7.28        2
A4   3.61   0.00   7.21        2
A5   7.07   3.61   6.71        2
A6   7.21   4.12   5.39        2
A7   8.06   7.21   0.00        3
A8   2.24   1.41   7.62        2
centroid ใหม่: [[2.0, 10.0], [6.0, 6.0], [1.5, 3.5]]
--- รอบที่ 2 ---
    d(C1)  d(C2)  d(C3)  cluster
A1   0.00   5.66   6.52        1
A2   5.00   4.12   1.58        3
A3   8.49   2.83   6.52        2
A4   3.61   2.24   5.70        2
A5   7.07   1.41   5.70        2
A6   7.21   2.00   4.53        2
A7   8.06   6.40   1.58        3
A8   2.24   3.61   6.04        1
centroid ใหม่: [[3.0, 9.5], [6.5, 5.25], [1.5, 3.5]]
sklearn labels   : [1 3 2 1 2 2 3 1]
sklearn centroids: [[3.667, 9.0], [7.0, 4.333], [1.5, 3.5]]
inertia: 14.333 | iterations: 4
```

</details>

### คำถามท้ายข้อ

1. จุดใดบ้างที่ย้ายกลุ่มระหว่างรอบที่ 1 กับรอบที่ 2 และเพราะเหตุใด
2. scikit-learn ใช้กี่รอบจึงลู่เข้า ผลสุดท้ายต่างจากผลหลังรอบที่ 2 ของเราอย่างไร (ดู A4)
3. จงแสดงการคำนวณ inertia สุดท้าย 14.333 จาก centroid สุดท้าย อย่างน้อยหนึ่งกลุ่ม

## ข้อที่ 2: แบ่งกลุ่มลูกค้า — Elbow, Silhouette และ Feature Scaling

### โจทย์

ห้างสรรพสินค้าแห่งหนึ่งมีข้อมูลสมาชิก 250 คน 3 feature ได้แก่ อายุ (`age`), รายได้ต่อเดือน (`income`, บาท) และคะแนนการใช้จ่าย (`spending`, 1–100) ต้องการแบ่งกลุ่มลูกค้าเพื่อออกแบบโปรโมชัน (ข้อมูลจำลองขึ้นจากลูกค้า 4 ประเภท แต่เราจะทำเหมือนไม่รู้จำนวนกลุ่ม)

1. สร้าง `X_scaled` ด้วย `StandardScaler`
2. สำหรับ k = 2 ถึง 7 ให้ fit `KMeans(n_clusters=k, n_init=10, random_state=42)` ทั้งกับ `X_raw` และ `X_scaled` บันทึก inertia และ silhouette score
3. วาดกราฟ Elbow และ Silhouette แล้วเลือก k ที่เหมาะสม
4. fit K-Means ด้วย k ที่เลือกบน `X_scaled` แล้วสรุปลักษณะแต่ละกลุ่มจากค่าเฉลี่ย

### โค้ดเริ่มต้น

```python
# สร้างข้อมูลลูกค้าจำลอง (ไม่ต้องแก้)
rng = np.random.default_rng(2569)
spec = {  # ชื่อกลุ่ม: (อายุเฉลี่ย, sd, รายได้เฉลี่ย, sd, คะแนนการใช้จ่าย, sd, จำนวนคน)
    "นักศึกษา":      (21, 2,  9000, 2000, 75,  8, 60),
    "วัยเริ่มทำงาน": (30, 3, 28000, 4000, 50, 10, 80),
    "ผู้บริหาร":     (45, 5, 60000, 8000, 45, 10, 40),
    "วัยเกษียณ":     (63, 4, 22000, 4000, 20,  8, 70),
}
parts = []
for a, sa, m, sm, s, ss, n in spec.values():
    parts.append(np.c_[rng.normal(a, sa, n), rng.normal(m, sm, n), np.clip(rng.normal(s, ss, n), 1, 100)])
customers = pd.DataFrame(np.vstack(parts).round(0), columns=["age", "income", "spending"])
customers = customers.sample(frac=1, random_state=0).reset_index(drop=True)   # สลับแถว ซ่อนเฉลย
print(customers.shape)
print(customers.head())
```

```python
X_raw = customers.values
X_scaled = X_raw.copy()   # TODO: แทนด้วย StandardScaler().fit_transform(customers)

results = []
for k in range(2, 8):
    # TODO: fit KMeans(n_clusters=k, n_init=10, random_state=42) กับ X_raw และกับ X_scaled
    #       แล้วเก็บ .inertia_ และ silhouette_score(X, model.labels_) ของทั้งสองแบบ
    inertia_raw, sil_raw = 0.0, 0.0
    inertia_scaled, sil_scaled = 0.0, 0.0
    results.append({"k": k,
                    "inertia_raw": inertia_raw, "sil_raw": sil_raw,
                    "inertia_scaled": inertia_scaled, "sil_scaled": sil_scaled})
res = pd.DataFrame(results).set_index("k")
print(res.round(3))
```

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(res.index, res["inertia_scaled"], "o-")
axes[0].set_title("Elbow (scaled)"); axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia")
axes[1].plot(res.index, res["sil_raw"], "o-", label="raw")
axes[1].plot(res.index, res["sil_scaled"], "s-", label="scaled")
axes[1].set_title("Silhouette"); axes[1].set_xlabel("k"); axes[1].legend()
plt.show()
```

```python
best_k = 2   # TODO: เปลี่ยนเป็นค่า k ที่เลือกจากกราฟ Elbow/Silhouette ของข้อมูลที่ scale แล้ว
km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X_scaled)
customers["cluster"] = km_best.labels_
print(customers.groupby("cluster").mean().round(0))
print(customers["cluster"].value_counts().sort_index())
```

### ผลลัพธ์ที่คาดหวัง

```output
(250, 3)
    age   income  spending
0  61.0  20847.0      14.0
1  31.0  25442.0      44.0
2  27.0  23579.0      54.0
3  47.0  43136.0      56.0
4  46.0  66847.0      44.0
    inertia_raw  sil_raw  inertia_scaled  sil_scaled
k                                                   
2  1.908141e+10    0.732         362.053       0.502
3  5.906203e+09    0.681         171.435       0.621
4  3.531349e+09    0.622          67.111       0.667
5  2.343866e+09    0.608          58.223       0.560
6  1.636655e+09    0.593          51.961       0.513
7  1.276854e+09    0.569          46.833       0.392
          age   income  spending
cluster                         
0        63.0  22035.0      20.0
1        30.0  28936.0      51.0
2        20.0   8885.0      77.0
3        46.0  60256.0      46.0
cluster
0    70
1    81
2    60
3    39
Name: count, dtype: int64
```

(ผลส่วนสุดท้ายเป็นของ `best_k = 4`)

### คำถามท้ายข้อ

1. ถ้าดูเฉพาะ silhouette ของข้อมูลที่ **ไม่ scale** จะเลือก k เท่าไร ทำไมผลนี้จึงหลอกเรา (ดูว่า feature ใดครอบงำระยะทาง)
2. กราฟ Elbow ของข้อมูลที่ scale แล้วหักศอกที่ k ใด สอดคล้องกับ silhouette หรือไม่
3. ตั้งชื่อลูกค้าแต่ละกลุ่มจากค่าเฉลี่ย และเสนอโปรโมชันที่เหมาะกับแต่ละกลุ่ม กลุ่มละ 1 แนวคิด

## ข้อที่ 3: DBSCAN กับ K-Means บนข้อมูลรูปพระจันทร์

### โจทย์

1. วัด ARI ของ K-Means (k = 2) บน `make_moons`
2. ลอง DBSCAN ด้วย `eps` = 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 (`min_samples=5`) บันทึกจำนวนกลุ่ม จำนวน noise และ ARI
3. เลือก `eps` ที่เหมาะสม แล้ววาดผลของ K-Means เทียบกับ DBSCAN

### โค้ดเริ่มต้น

```python
X3, y3 = make_moons(n_samples=600, noise=0.07, random_state=7)

km3 = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X3)
print("K-Means ARI:", round(adjusted_rand_score(y3, km3.labels_), 3))

for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    lab = np.zeros(len(X3), dtype=int)   # TODO: แทนด้วย DBSCAN(eps=eps, min_samples=5).fit_predict(X3)
    n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
    print(f"eps={eps:<4}: clusters={n_clusters:2d}, noise={(lab == -1).sum():3d}, ARI={adjusted_rand_score(y3, lab):.3f}")
```

```python
chosen_eps = 0.5   # TODO: เปลี่ยนเป็นค่า eps ที่เลือกจากตาราง
db3 = DBSCAN(eps=chosen_eps, min_samples=5).fit(X3)
print("core:", len(db3.core_sample_indices_), "| noise:", (db3.labels_ == -1).sum())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X3[:, 0], X3[:, 1], c=km3.labels_, s=10)
axes[0].set_title("K-Means (k=2)")
axes[1].scatter(X3[:, 0], X3[:, 1], c=db3.labels_, s=10)
axes[1].set_title(f"DBSCAN (eps={chosen_eps})")
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
K-Means ARI: 0.232
eps=0.05: clusters=35, noise=231, ARI=0.026
eps=0.1 : clusters= 2, noise=  4, ARI=0.987
eps=0.15: clusters= 2, noise=  2, ARI=0.993
eps=0.2 : clusters= 2, noise=  1, ARI=0.997
eps=0.3 : clusters= 2, noise=  0, ARI=1.000
eps=0.5 : clusters= 1, noise=  0, ARI=0.000
core: 598 | noise: 2
```

(บรรทัดสุดท้ายเป็นของ `chosen_eps = 0.15` — ค่า 0.1 ถึง 0.3 ใช้ได้ทั้งหมด)

### คำถามท้ายข้อ

1. ทำไม K-Means จึงได้ ARI ต่ำบนข้อมูลนี้ อธิบายโดยอ้างถึงรูปร่างของขอบเขตที่ K-Means สร้าง
2. อธิบายสิ่งที่เกิดขึ้นเมื่อ `eps` เล็กเกินไป (0.05) และใหญ่เกินไป (0.5) โดยใช้นิยาม core / border / noise
3. ในงานจริงที่ไม่มี `y3` ให้คำนวณ ARI เราจะเลือก `eps` อย่างไร

## ข้อที่ 4: PCA บนภาพตัวเลข (load_digits)

### โจทย์

ชุดข้อมูล `load_digits` มีภาพตัวเลขขนาด 8×8 = 64 พิกเซล (ค่า 0–16) จำนวน 1,797 ภาพ

1. หา explained variance ratio ของ 3 PC แรก และ variance สะสมที่ 2, 10, 20, 30 มิติ
2. หาจำนวนมิติที่เก็บ variance ได้ 95% แล้วลดมิติด้วย `PCA(n_components=0.95)`
3. สร้างภาพกลับจาก PCA 2, 10, 30 มิติด้วย `inverse_transform` วัด reconstruction MSE และแสดงภาพเทียบกับต้นฉบับ
4. เปรียบเทียบ ARI ของ K-Means (k = 10) บนข้อมูล 64 มิติ กับข้อมูลที่ลดมิติแล้ว

### โค้ดเริ่มต้น

```python
digits = load_digits()
X4, y4 = digits.data, digits.target
print(X4.shape, "ค่าพิกเซล", X4.min(), "-", X4.max())

pca_full = PCA().fit(X4)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
print("3 PC แรก:", pca_full.explained_variance_ratio_[:3].round(3))
for d in [2, 10, 20, 30]:
    print(f"d={d:2d}: variance สะสม = {cumsum[d-1]:.3f}")

n95 = 0                 # TODO: หาจำนวนมิติแรกที่ cumsum >= 0.95 (คำใบ้: np.argmax(cumsum >= 0.95) + 1)
print("จำนวนมิติที่เก็บ variance 95%:", n95)
X4_reduced = X4.copy()  # TODO: แทนด้วย PCA(n_components=0.95, random_state=42).fit_transform(X4)
print("shape:", X4.shape, "->", X4_reduced.shape)

plt.plot(np.arange(1, 65), cumsum)
plt.axhline(0.95, ls="--")
plt.xlabel("number of components"); plt.ylabel("cumulative explained variance")
plt.show()
```

```python
dims = [2, 10, 30]
fig, axes = plt.subplots(len(dims) + 1, 5, figsize=(8, 7))
for j in range(5):
    axes[0, j].imshow(X4[j].reshape(8, 8), cmap="gray_r"); axes[0, j].axis("off")
for i, d in enumerate(dims, start=1):
    pca_d = PCA(n_components=d, random_state=42)
    X_rec = X4.copy()   # TODO: ลดมิติด้วย pca_d.fit_transform(X4) แล้วแปลงกลับด้วย pca_d.inverse_transform(...)
    print(f"PCA {d:2d}D: reconstruction MSE = {np.mean((X4 - X_rec) ** 2):.3f}")
    for j in range(5):
        axes[i, j].imshow(X_rec[j].reshape(8, 8), cmap="gray_r"); axes[i, j].axis("off")
plt.show()

ari_raw = adjusted_rand_score(y4, KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(X4))
ari_pca = adjusted_rand_score(y4, KMeans(n_clusters=10, n_init=10, random_state=42).fit_predict(X4_reduced))
print(f"ARI 64D = {ari_raw:.3f} | ARI PCA {X4_reduced.shape[1]}D = {ari_pca:.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
(1797, 64) ค่าพิกเซล 0.0 - 16.0
3 PC แรก: [0.149 0.136 0.118]
d= 2: variance สะสม = 0.285
d=10: variance สะสม = 0.738
d=20: variance สะสม = 0.894
d=30: variance สะสม = 0.959
จำนวนมิติที่เก็บ variance 95%: 29
shape: (1797, 64) -> (1797, 29)
PCA  2D: reconstruction MSE = 13.421
PCA 10D: reconstruction MSE = 4.914
PCA 30D: reconstruction MSE = 0.768
ARI 64D = 0.667 | ARI PCA 29D = 0.662
```

### คำถามท้ายข้อ

1. PCA ลดข้อมูลจาก 64 เหลือ 29 มิติ (ลดลงกว่าครึ่ง) แต่ ARI ของ K-Means แทบไม่เปลี่ยน ข้อนี้บอกอะไรเกี่ยวกับพิกเซลที่ถูกตัดทิ้ง
2. ภาพที่สร้างกลับจาก 2 มิติมีลักษณะอย่างไร เปรียบเทียบกับ 30 มิติ
3. ทำไมใบงานนี้ไม่ใช้ t-SNE เพื่อลดมิติก่อนส่งให้โมเดลทำนายภาพใหม่

## ข้อที่ 5: ตรวจจับเครื่องจักรผิดปกติด้วย GMM และ Isolation Forest

### โจทย์

เซนเซอร์ของเครื่องจักรวัดอุณหภูมิ (`temp_C`) และการสั่นสะเทือน (`vibration_mm_s`) เครื่องทำงานปกติ 2 โหมด คือ idle (300 ครั้ง) และ running (400 ครั้ง) และมีเหตุขัดข้องปนอยู่ 10 ครั้ง (เฉลย `y5` ใช้ตรวจผลเท่านั้น)

1. scale ข้อมูล แล้ว fit `GaussianMixture(n_components=2)` (หนึ่ง component ต่อหนึ่งโหมดปกติ)
2. คำนวณ log-density ด้วย `score_samples` แล้วใช้ threshold ที่ percentile 1%, 2%, 5% เปรียบเทียบ precision/recall
3. ใช้ `IsolationForest(contamination=0.02)` แล้วเปรียบเทียบกับ GMM

### โค้ดเริ่มต้น

```python
rng = np.random.default_rng(7)
idle  = np.c_[rng.normal(35, 2, 300), rng.normal(0.5, 0.1, 300)]
run   = np.c_[rng.normal(70, 4, 400), rng.normal(2.0, 0.3, 400)]
fault = np.c_[rng.normal(90, 8, 10),  rng.normal(4.5, 0.8, 10)]
sensor = pd.DataFrame(np.vstack([idle, run, fault]), columns=["temp_C", "vibration_mm_s"])
y5 = np.r_[np.zeros(700, dtype=int), np.ones(10, dtype=int)]   # เฉลย ใช้ตรวจผลเท่านั้น
X5 = StandardScaler().fit_transform(sensor)
print(sensor.describe().round(2))

gm5 = GaussianMixture(n_components=2, n_init=10, random_state=42).fit(X5)
dens = np.zeros(len(X5))   # TODO: แทนด้วย gm5.score_samples(X5)
for pct in [1, 2, 5]:
    thr = np.percentile(dens, pct)
    pred = (dens < thr).astype(int)
    print(f"GMM pct={pct}%: thr={thr:7.2f} flagged={pred.sum():3d} "
          f"precision={precision_score(y5, pred, zero_division=0):.2f} recall={recall_score(y5, pred):.2f}")

iso5 = IsolationForest(contamination=0.02, random_state=42).fit(X5)
pred_iso = np.zeros(len(X5), dtype=int)   # TODO: แปลง iso5.predict(X5) (-1 = ผิดปกติ) เป็น 1/0
print(f"IsolationForest: flagged={pred_iso.sum()} "
      f"precision={precision_score(y5, pred_iso, zero_division=0):.2f} recall={recall_score(y5, pred_iso):.2f}")

plt.scatter(sensor["temp_C"], sensor["vibration_mm_s"], c=dens, s=10, cmap="viridis")
plt.colorbar(label="log-density"); plt.xlabel("temp_C"); plt.ylabel("vibration_mm_s")
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
       temp_C  vibration_mm_s
count  710.00          710.00
mean    55.39            1.39
std     18.11            0.87
min     28.50            0.22
25%     35.09            0.50
50%     66.13            1.65
75%     70.70            2.06
max    105.45            5.81
GMM pct=1%: thr=  -8.17 flagged=  8 precision=1.00 recall=0.80
GMM pct=2%: thr=  -3.42 flagged= 15 precision=0.67 recall=1.00
GMM pct=5%: thr=  -2.38 flagged= 36 precision=0.28 recall=1.00
IsolationForest: flagged=15 precision=0.67 recall=1.00
```

### คำถามท้ายข้อ

1. เมื่อเพิ่ม percentile จาก 1% เป็น 5% precision และ recall เปลี่ยนไปอย่างไร เพราะอะไร
2. ถ้าการพลาดเหตุขัดข้องจริงทำให้เครื่องจักรพังและเสียหายหลักแสนบาท ขณะที่การแจ้งเตือนผิดเสียแค่ค่าตรวจเช็ก ควรเลือก threshold ใด
3. ทำไมจึงใช้ `n_components=2` แทนที่จะใช้ Gaussian ตัวเดียว ถ้าใช้ K-Means แทน GMM จะได้ข้อมูลอะไรน้อยลง

## ข้อที่ 6: วิเคราะห์กรณีศึกษา — เลือกอัลกอริทึมให้เหมาะสม

### โจทย์

สำหรับแต่ละสถานการณ์ ให้ระบุ (ก) ประเภทงาน (clustering / dimensionality reduction / anomaly detection / supervised) (ข) อัลกอริทึมที่เลือก (ค) ขั้นตอน preprocessing ที่จำเป็น และ (ง) วิธีประเมินผล พร้อมเหตุผลสั้น ๆ ตอบในเซลล์ markdown ของ notebook

**กรณี 1:** ร้านกาแฟเครือข่ายมีข้อมูลสมาชิก 50,000 คน (ความถี่การซื้อ ยอดซื้อเฉลี่ย ช่วงเวลาที่ซื้อ) ต้องการแบ่งลูกค้าเป็น 4–6 กลุ่มเพื่อส่งคูปองต่างกัน และต้องจัดกลุ่มสมาชิกใหม่ทุกสัปดาห์

<details><summary>แนวคำตอบ</summary>

(ก) Clustering (ข) K-Means เพราะต้องการจำนวนกลุ่มที่ควบคุมได้ เร็วกับข้อมูลหลายหมื่นแถว ตีความผ่าน centroid ได้ และมี `predict()` สำหรับสมาชิกใหม่ (ค) StandardScaler ใส่ใน Pipeline (ง) Elbow + Silhouette สำหรับ k = 4–6 ร่วมกับการตรวจว่าแต่ละกลุ่มมีความหมายทางการตลาด

</details>

**กรณี 2:** หน่วยงานจราจรมีพิกัด GPS ของจุดเกิดอุบัติเหตุ 3 ปี ต้องการหา "จุดเสี่ยง" ซึ่งมักเป็นแนวยาวตามถนน ไม่ทราบว่ามีกี่จุด และมีอุบัติเหตุกระจัดกระจายที่ไม่ใช่จุดเสี่ยงจำนวนมาก

<details><summary>แนวคำตอบ</summary>

(ก) Clustering (ข) DBSCAN เพราะไม่ต้องกำหนด k ไล่ตามรูปร่างกลุ่มที่ไม่ใช่ทรงกลม และแยกอุบัติเหตุกระจัดกระจายเป็น noise ได้ (ค) แปลงพิกัดเป็นหน่วยระยะทางเดียวกัน (เช่น เมตร) เพื่อให้ eps มีความหมาย (ง) ลองหลายค่า eps/min_samples ตรวจจำนวนกลุ่มและ noise และให้ผู้เชี่ยวชาญตรวจบนแผนที่

</details>

**กรณี 3:** โรงงานติดเซนเซอร์ 300 ช่องบนสายการผลิต ต้องการ (1) ลดขนาดข้อมูลก่อนส่งเข้าโมเดลพยากรณ์ของเสีย และ (2) วาดภาพรวมให้วิศวกรเห็นว่าข้อมูลแต่ละกะต่างกันหรือไม่

<details><summary>แนวคำตอบ</summary>

(1) Dimensionality reduction ด้วย PCA (`n_components=0.95`) หลัง StandardScaler เพราะเป็นเชิงเส้น เร็ว และมี `transform()` สำหรับข้อมูลใหม่ ประเมินจาก explained variance และ performance ของโมเดลปลายทาง (2) PCA ลดเหลือ 30–50 มิติก่อน แล้วใช้ t-SNE ลดเหลือ 2 มิติเพื่อวาดกราฟระบายสีตามกะ ใช้เพื่อดูเท่านั้น ไม่ใส่ใน pipeline การทำนาย

</details>

**กรณี 4:** ธนาคารต้องการตรวจจับธุรกรรมบัตรเครดิตที่น่าสงสัยแบบ real-time มีธุรกรรมวันละหลายแสนรายการ ธุรกรรมโกงที่ยืนยันแล้วมีน้อยมาก และรูปแบบการโกงเปลี่ยนไปเรื่อย ๆ

<details><summary>แนวคำตอบ</summary>

(ก) Anomaly detection (ข) Isolation Forest (เร็ว รองรับหลาย feature ไม่ต้องสมมติการแจกแจง) หรือ GMM + density threshold ถ้า feature ไม่มากและข้อมูลปกติเป็นกลุ่มวงรี (ค) scale feature และ encode feature เชิงหมวดหมู่ (ง) ใช้ธุรกรรมโกงที่ยืนยันแล้วจำนวนน้อยวัด precision/recall และเลือก threshold ตามต้นทุนของการพลาด (มักยอมให้ recall สูง) — เมื่อมี label มากพอในอนาคตอาจเสริมด้วยโมเดล supervised

</details>

**กรณี 5:** นักศึกษาคนหนึ่งใช้ K-Means กับข้อมูลผู้ป่วยที่มี feature ความดันโลหิต (mmHg, 90–180) และจำนวนเม็ดเลือดขาว (เซลล์/µL, 4,000–11,000) โดยไม่ scale แล้วสรุปว่า "ความดันโลหิตไม่มีผลต่อการแบ่งกลุ่ม" ข้อสรุปนี้ถูกต้องหรือไม่

<details><summary>แนวคำตอบ</summary>

ไม่ถูกต้อง เพราะ K-Means ใช้ระยะ Euclidean จำนวนเม็ดเลือดขาวมีช่วงค่าใหญ่กว่าความดันหลายสิบเท่า จึงครอบงำระยะทางและกลบผลของความดัน ต้อง StandardScaler ก่อนแล้วจึงสรุปใหม่ (เหมือนผล silhouette ของข้อมูลที่ไม่ scale ในข้อ 2 ซึ่งชี้ k ผิด)

</details>

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | คะแนน | สิ่งที่ตรวจในใบงานนี้ |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | 2 | อธิบาย inertia, silhouette, core/border/noise, explained variance, log-density ได้ถูกต้องในคำถามท้ายข้อ |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | 2 | เลือก k, eps, threshold อย่างมีเหตุผล และตอบกรณีศึกษาข้อ 6 ได้เหมาะสม |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | 2 | ตารางคำนวณมือข้อ 1 ครบ 2 รอบ และ pipeline scale → model → evaluate ครบในข้อ 2–5 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | 3 | TODO ทุกจุดถูกต้อง notebook รันได้ตั้งแต่ต้นจนจบ ผลลัพธ์ตรงกับที่คาดหวัง |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | 1 | ตีความกลุ่มลูกค้า ผลของ scaling และข้อจำกัดของ K-Means / t-SNE / threshold |
| **รวม** | **10** | |

## สิ่งที่ต้องส่ง

1. ไฟล์ `lab-05.ipynb` ที่รันครบทุกเซลล์ (Runtime → Run all) และแสดงผลลัพธ์ทุกเซลล์ ตั้งชื่อเป็น `lab05_รหัสนักศึกษา.ipynb`
2. ตารางคำนวณ K-Means ด้วยมือรอบที่ 1–2 ของข้อ 1 (พิมพ์ในเซลล์ markdown หรือแนบภาพถ่ายลายมือ)
3. คำตอบคำถามท้ายข้อ 1–5 และคำตอบกรณีศึกษาข้อ 6 ในเซลล์ markdown ใต้แต่ละข้อ
4. ส่งผ่านระบบ LMS ของรายวิชาภายในวันที่ผู้สอนกำหนด
