## วัตถุประสงค์การเรียนรู้

- แยกแยะปัญหา binary, multiclass และ multilabel classification และอธิบายหลักการของ Logistic Regression (sigmoid, threshold, softmax) ได้ (LLO2-2)
- อธิบายแนวคิด จุดเด่น และข้อจำกัดของ KNN, SVM, Decision Tree และ Ensemble (Voting, Bagging, Random Forest, Boosting) แล้วเลือกอัลกอริทึมให้เหมาะกับลักษณะข้อมูลได้ (LLO2-2)
- อ่าน confusion matrix ในรูปแบบของ scikit-learn และ**คำนวณ accuracy, precision, recall, F1-score ด้วยมือ**ได้ถูกต้อง (LLO2-3)
- อธิบายว่าทำไม accuracy จึงหลอกเราได้เมื่อข้อมูลไม่สมดุล และปรับ decision threshold เพื่อแลก precision กับ recall ได้ (LLO2-3)
- พัฒนาโมเดล classification ด้วย Python/scikit-learn บนข้อมูลจริง (Breast Cancer, MNIST) และตีความ `classification_report` ได้ (LLO2-3)

## 1. Classification คืออะไร

ใน [บทที่ 1](/module/01) เราแบ่ง Supervised Learning ออกเป็น 2 กลุ่มใหญ่ตาม "ชนิดของคำตอบ" ที่ต้องการทำนาย

- **Regression** — คำตอบเป็นตัวเลขต่อเนื่อง ("เท่าไหร่?") เช่น ราคาบ้าน อุณหภูมิพรุ่งนี้ (เรียนใน [บทที่ 4](/module/04))
- **Classification** — คำตอบเป็นหมวดหมู่ ("แบบไหน?") เช่น อีเมลนี้เป็น spam หรือไม่ ภาพนี้คือเลขอะไร คนไข้รายนี้เสี่ยงเบาหวานหรือไม่

![เปรียบเทียบ Classification (ทำนายหมวดหมู่) กับ Regression (ทำนายค่าต่อเนื่อง)](/images/m01/compare.jpg)

โมเดล classification พยายามลาก **เส้นแบ่งเขตแดน (decision boundary)** ใน feature space เพื่อแยกข้อมูลแต่ละคลาสออกจากกัน อัลกอริทึมแต่ละตัวต่างกันที่ "รูปร่าง" ของเส้นแบ่งที่ลากได้ เช่น Logistic Regression ลากได้เฉพาะเส้นตรง (hyperplane) ส่วน Decision Tree ลากเป็นขั้นบันไดตั้งฉากกับแกน และ KNN ลากเส้นโค้งไปตามข้อมูลได้อิสระ

### 1.1 ประเภทของปัญหา Classification

| ประเภท | ความหมาย | ตัวอย่าง |
|---|---|---|
| **Binary** | มี 2 คลาส (มักเรียก positive = 1 / negative = 0) | ป่วย/ไม่ป่วย, spam/ไม่ใช่ spam, ทุจริต/ปกติ |
| **Multiclass** | มีมากกว่า 2 คลาส แต่ละตัวอย่างอยู่ได้คลาสเดียว | ตัวเลข 0–9 (MNIST), พันธุ์ดอกไม้ 3 พันธุ์ (Iris) |
| **Multilabel** | หนึ่งตัวอย่างมีได้หลาย label พร้อมกัน | ภาพหนึ่งภาพมีทั้ง "แมว" และ "สุนัข", บทความมีหลายแท็ก |

> [!NOTE]
> **คลาส positive ไม่ได้แปลว่า "ดี"** แต่หมายถึง "สิ่งที่เรากำลังตามหา" เช่น ในงานคัดกรองโรค positive = ป่วย ในงานตรวจจับการโกง positive = ธุรกรรมทุจริต การกำหนดว่าคลาสไหนเป็น positive สำคัญมาก เพราะ precision และ recall คำนวณจากมุมมองของคลาส positive

อัลกอริทึมบางตัว (Logistic Regression แบบ softmax, KNN, Decision Tree, Random Forest) รองรับหลายคลาสได้โดยตรง ส่วนตัวที่ออกแบบมาสำหรับ 2 คลาส (เช่น SVM) จะใช้กลยุทธ์ **One-vs-Rest (OvR)** — สร้างตัวจำแนก 1 ตัวต่อ 1 คลาส ("เลข 3 หรือไม่ใช่ 3") หรือ **One-vs-One (OvO)** — สร้างตัวจำแนกทุกคู่คลาส ($N(N-1)/2$ ตัว เช่น MNIST 10 คลาส = 45 ตัว) scikit-learn เลือกกลยุทธ์ให้อัตโนมัติ เช่น `SVC` ใช้ OvO ภายใน

### 1.2 รูปแบบการใช้งานใน scikit-learn

ตัวจำแนก (classifier) ทุกตัวใน scikit-learn มี interface เดียวกัน ทำให้สลับอัลกอริทึมได้ง่าย

| เมธอด | หน้าที่ |
|---|---|
| `fit(X_train, y_train)` | ฝึกโมเดล |
| `predict(X)` | ทำนายคลาส |
| `predict_proba(X)` | ทำนายความน่าจะเป็นของแต่ละคลาส (ไม่ใช่ทุกโมเดลจะมี) |
| `decision_function(X)` | คะแนนดิบก่อนตัดสินใจ (เช่น ระยะจากเส้นแบ่งของ SVM) |
| `score(X, y)` | คืนค่า accuracy |

## 2. Logistic Regression

ชื่อมีคำว่า "Regression" แต่ใช้สำหรับ **classification** แนวคิดคือนำสมการเชิงเส้นแบบเดียวกับ Linear Regression มาครอบด้วยฟังก์ชัน **sigmoid (logistic function)** เพื่อบีบผลลัพธ์ให้อยู่ในช่วง 0 ถึง 1 แล้วตีความเป็น **ความน่าจะเป็นที่จะเป็นคลาส positive**

$$ z = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n = \boldsymbol{\theta}^T\mathbf{x} $$

$$ \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}} $$

ฟังก์ชัน sigmoid มีคุณสมบัติที่ควรจำ: $\sigma(0) = 0.5$, เมื่อ $z \to +\infty$ ค่าเข้าใกล้ 1 และเมื่อ $z \to -\infty$ ค่าเข้าใกล้ 0

### 2.1 Decision threshold

เมื่อได้ความน่าจะเป็น $\hat{p}$ แล้ว โมเดลตัดสินคลาสด้วย **threshold** ซึ่งค่าเริ่มต้นคือ 0.5

$$ \hat{y} = \begin{cases} 1 & \text{ถ้า } \hat{p} \ge 0.5 \;(\text{เทียบเท่ากับ } z \ge 0) \\ 0 & \text{ถ้า } \hat{p} < 0.5 \end{cases} $$

ดังนั้น decision boundary ของ Logistic Regression คือเซตของจุดที่ $z = 0$ ซึ่งเป็นเส้นตรง (หรือ hyperplane) เสมอ การฝึกโมเดลคือการหา $\boldsymbol{\theta}$ ที่ทำให้ **log loss (cross-entropy)** ต่ำที่สุด

$$ J(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] $$

ตัวอย่าง: ทำนายว่าดอก Iris เป็นพันธุ์ *virginica* หรือไม่ จากความกว้างกลีบดอก (petal width) เพียงฟีเจอร์เดียว

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
X = iris.data[["petal width (cm)"]].values          # ใช้ฟีเจอร์เดียว: ความกว้างกลีบดอก
y = (iris.target == 2).astype(int).values           # 1 = Iris virginica, 0 = ไม่ใช่ (แทน np.int ที่ถูกถอดออกแล้ว)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

X_new = np.array([[1.0], [1.6], [1.7], [2.5]])
print("P(ไม่ใช่, ใช่):\n", log_reg.predict_proba(X_new).round(3))
print("predict:", log_reg.predict(X_new))

# คำนวณเองด้วย sigmoid เพื่อยืนยัน
z = log_reg.intercept_[0] + log_reg.coef_[0, 0] * X_new[:, 0]
print("sigmoid(z):", (1 / (1 + np.exp(-z))).round(3))

# decision boundary: จุดที่ z = 0 -> x = -b/w
print(f"decision boundary ≈ {-log_reg.intercept_[0] / log_reg.coef_[0, 0]:.2f} cm")
```

```output
P(ไม่ใช่, ใช่):
 [[0.946 0.054]
 [0.565 0.435]
 [0.457 0.543]
 [0.026 0.974]]
predict: [0 0 1 1]
sigmoid(z): [0.054 0.435 0.543 0.974]
decision boundary ≈ 1.66 cm
```

**การตีความ:** `predict_proba` คืนค่า 2 คอลัมน์ คือ P(คลาส 0) และ P(คลาส 1) ซึ่งรวมกันได้ 1 เสมอ ค่าในคอลัมน์ที่สองตรงกับ sigmoid ที่เราคำนวณเอง ดอกที่กลีบกว้าง 1.6 cm มีโอกาสเป็น virginica 43.5% (< 0.5 จึงทายว่า 0) ส่วน 1.7 cm มีโอกาส 54.3% (ทายว่า 1) เส้นแบ่งจึงอยู่ที่ราว 1.66 cm

> [!TIP]
> ค่า `predict_proba` มีประโยชน์มากกว่าแค่ `predict` เพราะเราเลื่อน threshold ได้เอง เช่น งานคัดกรองโรคอาจใช้ threshold 0.3 เพื่อให้จับผู้ป่วยได้มากขึ้น (ดูหัวข้อ 7.6)

### 2.2 Softmax Regression สำหรับหลายคลาส

เมื่อมี $K$ คลาส Logistic Regression จะคำนวณคะแนน $s_k(\mathbf{x}) = \boldsymbol{\theta}^{(k)T}\mathbf{x}$ สำหรับแต่ละคลาส แล้วแปลงเป็นความน่าจะเป็นด้วยฟังก์ชัน **softmax**

$$ \hat{p}_k = \frac{e^{s_k(\mathbf{x})}}{\sum_{j=1}^{K} e^{s_j(\mathbf{x})}} $$

ความน่าจะเป็นของทุกคลาสรวมกันได้ 1 และโมเดลเลือกคลาสที่ $\hat{p}_k$ สูงที่สุด

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target.values                               # 3 คลาส: 0=setosa, 1=versicolor, 2=virginica

softmax_reg = LogisticRegression(C=30, max_iter=1000, random_state=42)
softmax_reg.fit(X, y)                                # หลายคลาส -> ใช้ softmax (multinomial) อัตโนมัติ

print("predict:", softmax_reg.predict([[5, 2]]))
print("predict_proba:", softmax_reg.predict_proba([[5, 2]]).round(3))
print("ผลรวมความน่าจะเป็น =", softmax_reg.predict_proba([[5, 2]]).sum())
```

```output
predict: [2]
predict_proba: [[0.    0.029 0.971]]
ผลรวมความน่าจะเป็น = 1.0
```

ดอกที่กลีบยาว 5 cm กว้าง 2 cm ถูกทายว่าเป็น virginica ด้วยความน่าจะเป็น 97.1%

> [!WARNING]
> โค้ดเก่าในอินเทอร์เน็ต (และในตำราฉบับก่อน) มักเขียน `LogisticRegression(multi_class="multinomial")` พารามิเตอร์ `multi_class` **ถูกถอดออกแล้ว**ใน scikit-learn รุ่นใหม่ — ปัจจุบันเมื่อ `y` มีมากกว่า 2 คลาส โมเดลจะใช้ softmax ให้อัตโนมัติ ไม่ต้องระบุอะไรเพิ่ม เช่นเดียวกับ `np.int` ที่ต้องเปลี่ยนเป็น `int`

พารามิเตอร์ `C` คือ **ค่าผกผันของความแรงของ regularization** — `C` น้อย = regularize มาก (โมเดลเรียบง่าย) `C` มาก = regularize น้อย (โมเดลยืดหยุ่นตามข้อมูลมากขึ้น)

## 3. K-Nearest Neighbors (KNN)

KNN เป็นอัลกอริทึมแบบ **instance-based** ที่เข้าใจง่ายที่สุด: "บอกฉันมาว่าเพื่อนบ้านของคุณคือใคร แล้วฉันจะบอกว่าคุณเป็นใคร"

1. เก็บข้อมูลฝึกทั้งหมดไว้ (การ `fit` แทบไม่ทำอะไรเลย)
2. เมื่อมีข้อมูลใหม่ คำนวณระยะห่างไปยังข้อมูลฝึก**ทุกตัว** โดยทั่วไปใช้ระยะแบบยุคลิด (Euclidean distance)

$$ d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{n}(x_j - x'_j)^2} $$

3. เลือก $k$ ตัวที่ใกล้ที่สุด แล้วโหวต คลาสที่ได้เสียงมากที่สุดคือคำตอบ

พารามิเตอร์สำคัญ

| พารามิเตอร์ | ผล |
|---|---|
| `n_neighbors` ($k$) | $k$ น้อย (เช่น 1) → เส้นแบ่งขรุขระ ไวต่อ noise เสี่ยง overfit / $k$ มาก → เส้นแบ่งเรียบ เสี่ยง underfit นิยมใช้ $k$ เป็นเลขคี่ในงาน binary เพื่อเลี่ยงคะแนนเสมอ |
| `weights` | `"uniform"` ทุกเพื่อนบ้านมีเสียงเท่ากัน / `"distance"` เพื่อนบ้านที่ใกล้กว่ามีน้ำหนักมากกว่า (น้ำหนัก = 1/ระยะ) |
| `metric` | วิธีวัดระยะ ค่าเริ่มต้นคือ Minkowski ที่ p=2 (= Euclidean) |

เพราะ KNN ตัดสินจาก "ระยะ" ฟีเจอร์ที่มีหน่วยใหญ่ (เช่น พื้นที่เนื้องอกหลักร้อยถึงหลักพัน) จะกลบฟีเจอร์ที่มีหน่วยเล็ก (เช่น ความเรียบของผิว 0.05–0.16) ดังนั้น **ต้อง scale ข้อมูลก่อนเสมอ** ลองดูผลบนข้อมูลมะเร็งเต้านม (Breast Cancer Wisconsin, 569 ตัวอย่าง 30 ฟีเจอร์)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)          # 569 ตัวอย่าง 30 ฟีเจอร์
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

for k in [1, 3, 5, 15, 51]:
    raw = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    scaled = make_pipeline(StandardScaler(),
                           KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
    print(f"k={k:>2} | ไม่ scale: {raw.score(X_test, y_test):.3f} | "
          f"StandardScaler: {scaled.score(X_test, y_test):.3f}")
```

```output
k= 1 | ไม่ scale: 0.921 | StandardScaler: 0.939
k= 3 | ไม่ scale: 0.930 | StandardScaler: 0.982
k= 5 | ไม่ scale: 0.912 | StandardScaler: 0.956
k=15 | ไม่ scale: 0.930 | StandardScaler: 0.974
k=51 | ไม่ scale: 0.930 | StandardScaler: 0.947
```

**การตีความ:** ทุกค่า $k$ การ scale ทำให้ accuracy ดีขึ้น และ $k=1$ ไม่ได้ดีที่สุดแม้จะ "จำ" ข้อมูลฝึกได้ 100% (test set มีเพียง 114 ตัวอย่าง ต่างกัน 1 ตัวอย่าง ≈ 0.009 จึงไม่ควรตีความความต่างเล็ก ๆ มากเกินไป — ในงานจริงให้เลือก $k$ ด้วย cross-validation)

**ข้อดี:** เข้าใจง่าย ไม่มีสมมติฐานเรื่องรูปร่างของข้อมูล เส้นแบ่งยืดหยุ่น **ข้อเสีย:** `predict` ช้ามากเมื่อข้อมูลฝึกเยอะ (ต้องวัดระยะกับทุกตัว) ใช้หน่วยความจำมาก และแย่ลงเมื่อมิติสูง (curse of dimensionality)

## 4. Support Vector Machines (SVM)

### 4.1 Large margin classification

SVM พยายามสร้าง "ถนน" ที่**กว้างที่สุด**คั่นระหว่างสองคลาส เส้นแบ่งคือเส้นกึ่งกลางถนน ข้อมูลที่อยู่บนขอบถนนเรียกว่า **support vectors** — มีเพียงจุดเหล่านี้เท่านั้นที่กำหนดเส้นแบ่ง การเพิ่มข้อมูลที่อยู่ไกลจากถนนไม่ทำให้เส้นแบ่งเปลี่ยน

- **Hard margin:** ห้ามมีข้อมูลอยู่ในถนนหรือผิดฝั่งเลย ใช้ได้เฉพาะข้อมูลที่แยกกันได้ด้วยเส้นตรงสมบูรณ์ และอ่อนไหวต่อ outlier มาก
- **Soft margin:** ยอมให้มีข้อมูลล้ำเข้ามาในถนนได้บ้าง ควบคุมด้วยพารามิเตอร์ `C`
  - `C` **น้อย** → ถนนกว้าง ยอมผิดได้มาก → regularize มาก ลด overfitting (แต่ถ้าน้อยเกินจะ underfit)
  - `C` **มาก** → ถนนแคบ พยายามทำให้ถูกทุกจุด → เสี่ยง overfitting

### 4.2 Kernel trick

ถ้าข้อมูลแยกด้วยเส้นตรงไม่ได้ วิธีหนึ่งคือเพิ่มฟีเจอร์ยกกำลัง (polynomial features) ให้ข้อมูลไปอยู่ในมิติที่สูงขึ้นซึ่งแยกด้วย hyperplane ได้ แต่ถ้าเพิ่มจริงจำนวนฟีเจอร์จะระเบิด **kernel trick** คือเทคนิคทางคณิตศาสตร์ที่ให้ผลลัพธ์ "เสมือน" ได้เพิ่มฟีเจอร์มิติสูงไปแล้ว โดยไม่ต้องสร้างฟีเจอร์เหล่านั้นจริง kernel ที่ใช้บ่อยคือ `"linear"`, `"poly"` และ `"rbf"` (Gaussian RBF ค่าเริ่มต้นของ `SVC`)

> [!WARNING]
> SVM **อ่อนไหวต่อสเกลของข้อมูลมาก** เพราะความกว้างของถนนวัดเป็นระยะ ต้องใช้ `StandardScaler` ก่อนเสมอ (แนะนำให้ใส่ไว้ใน `Pipeline`)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "SVC rbf (ไม่ scale)":       SVC(kernel="rbf", C=1),
    "LinearSVC + scaler":        make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42)),
    "SVC rbf + scaler, C=0.01":  make_pipeline(StandardScaler(), SVC(kernel="rbf", C=0.01)),
    "SVC rbf + scaler, C=1":     make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1)),
    "SVC rbf + scaler, C=100":   make_pipeline(StandardScaler(), SVC(kernel="rbf", C=100)),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:<26} train={model.score(X_train, y_train):.3f}  test={model.score(X_test, y_test):.3f}")

svc = models["SVC rbf + scaler, C=1"][-1]
print("จำนวน support vectors ต่อคลาส:", svc.n_support_)
```

```output
SVC rbf (ไม่ scale)        train=0.919  test=0.930
LinearSVC + scaler         train=0.989  test=0.965
SVC rbf + scaler, C=0.01   train=0.626  test=0.632
SVC rbf + scaler, C=1      train=0.982  test=0.982
SVC rbf + scaler, C=100    train=1.000  test=0.947
```

```output
จำนวน support vectors ต่อคลาส: [51 46]
```

**การตีความ:** เห็นทั้งสามเรื่องในตารางเดียว (1) ไม่ scale ได้แค่ 0.930 (2) `C=0.01` regularize มากเกินจน **underfit** (train ก็ต่ำ — ทายเป็นคลาสส่วนใหญ่เกือบทั้งหมด) (3) `C=100` ได้ train = 1.000 แต่ test ตกลง เป็นสัญญาณของ **overfitting** ค่า `C=1` สมดุลที่สุดในกรณีนี้ และมีเพียง 97 จุดจาก 455 จุดที่เป็น support vectors

**ข้อดี:** แม่นยำสูงกับข้อมูลขนาดเล็ก–กลางที่มีมิติสูง เส้นแบ่งไม่เชิงเส้นได้ด้วย kernel **ข้อเสีย:** ฝึกช้าเมื่อข้อมูลเกินหลักแสนแถว ต้อง scale ต้องจูน `C`/`gamma` และ `SVC` ไม่ได้ให้ความน่าจะเป็นโดยตรง (ใช้ `decision_function` แทน)

## 5. Decision Trees

### 5.1 แนวคิด

Decision Tree ตัดสินใจด้วยการถามคำถามแบบใช่/ไม่ใช่ต่อกันเป็นทอด ๆ จาก **root node** ลงไปยัง **child node** จนถึง **leaf node** ซึ่งให้คำตอบ เป็นโมเดลแบบ **white-box** — มนุษย์อ่านและอธิบายเหตุผลได้ทุกขั้น (ต่างจาก neural network ที่เป็น black-box) และ**ไม่ต้อง scale ข้อมูล** เพราะแต่ละ node เปรียบเทียบฟีเจอร์ทีละตัวกับค่าเกณฑ์

### 5.2 อัลกอริทึม CART และการวัดความบริสุทธิ์

scikit-learn ใช้อัลกอริทึม **CART** (Classification And Regression Tree) ซึ่งสร้างต้นไม้แบบไบนารี: ที่แต่ละ node จะค้นหาคู่ (ฟีเจอร์ $k$, ค่าเกณฑ์ $t_k$) ที่แบ่งข้อมูลออกเป็น 2 กลุ่มให้ "บริสุทธิ์" (pure) ที่สุด โดยลด cost function นี้ให้ต่ำสุด

$$ J(k, t_k) = \frac{m_{\text{left}}}{m}G_{\text{left}} + \frac{m_{\text{right}}}{m}G_{\text{right}} $$

ความไม่บริสุทธิ์ $G$ วัดได้ 2 แบบ ($p_{i,k}$ = สัดส่วนของคลาส $k$ ใน node $i$)

$$ \text{Gini}_i = 1 - \sum_{k=1}^{K} p_{i,k}^2 \qquad\qquad H_i = -\sum_{k=1}^{K} p_{i,k}\log_2 p_{i,k} $$

ทั้งสองค่ายิ่งต่ำยิ่งบริสุทธิ์ node ที่มีคลาสเดียวล้วนมี Gini = 0 และ entropy = 0 ในทางปฏิบัติผลลัพธ์ของทั้งสองใกล้เคียงกันมาก Gini คำนวณเร็วกว่าเล็กน้อยจึงเป็นค่าเริ่มต้น (`criterion="gini"`)

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target.values

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

print(export_text(tree_clf,
                  feature_names=["petal_length", "petal_width"],
                  show_weights=True))
print("predict_proba([5, 1.5]):", tree_clf.predict_proba([[5, 1.5]]).round(3))
print("predict([5, 1.5]):", iris.target_names[tree_clf.predict([[5, 1.5]])])
```

```output
|--- petal_length <= 2.45
|   |--- weights: [50.00, 0.00, 0.00] class: 0
|--- petal_length >  2.45
|   |--- petal_width <= 1.75
|   |   |--- weights: [0.00, 49.00, 5.00] class: 1
|   |--- petal_width >  1.75
|   |   |--- weights: [0.00, 1.00, 45.00] class: 2

predict_proba([5, 1.5]): [[0.    0.907 0.093]]
predict([5, 1.5]): ['versicolor']
```

**คำนวณ Gini ด้วยมือ** (`weights` คือจำนวนตัวอย่างของคลาส setosa, versicolor, virginica ใน node นั้น)

- Root node มี [50, 50, 50]: $\text{Gini} = 1 - (1/3)^2 - (1/3)^2 - (1/3)^2 = 0.667$
- Leaf ซ้าย [50, 0, 0]: $\text{Gini} = 1 - 1^2 = 0$ (บริสุทธิ์สมบูรณ์)
- Leaf กลาง [0, 49, 5]: $\text{Gini} = 1 - (49/54)^2 - (5/54)^2 = 0.168$
- Cost ของการแบ่งที่ root: node ขวามี [0, 50, 50] → Gini = 0.5 ดังนั้น $J = \frac{50}{150}(0) + \frac{100}{150}(0.5) = 0.333$ ลดลงจาก 0.667

ดอกที่ petal length = 5, petal width = 1.5 ตกที่ leaf กลาง จึงได้ `predict_proba` = [0/54, 49/54, 5/54] = [0, 0.907, 0.093] — ความน่าจะเป็นของต้นไม้คือ**สัดส่วนของคลาสใน leaf** นั่นเอง

> [!TIP]
> ใน Colab ใช้ `from sklearn.tree import plot_tree` แล้ว `plot_tree(tree_clf, filled=True, feature_names=[...], class_names=iris.target_names)` เพื่อวาดต้นไม้เป็นภาพ — เหมาะสำหรับอธิบายโมเดลให้ผู้ใช้ที่ไม่ใช่โปรแกรมเมอร์ เช่น แพทย์ หรือผู้บริหาร

### 5.3 การป้องกัน Overfitting

ถ้าไม่จำกัด ต้นไม้จะแตกกิ่งไปเรื่อย ๆ จนทุก leaf บริสุทธิ์ = ท่องจำข้อมูลฝึก (train accuracy 100%) จึงต้อง regularize ด้วย hyperparameter

| พารามิเตอร์ | ความหมาย |
|---|---|
| `max_depth` | ความลึกสูงสุดของต้นไม้ |
| `min_samples_split` | จำนวนตัวอย่างขั้นต่ำใน node ก่อนจะแบ่งต่อได้ |
| `min_samples_leaf` | จำนวนตัวอย่างขั้นต่ำที่ต้องมีใน leaf |
| `max_leaf_nodes` | จำนวน leaf สูงสุด |

หลักจำ: **เพิ่ม `min_*` หรือลด `max_*` = regularize มากขึ้น** ข้อเสียอีกข้อของต้นไม้เดี่ยวคือ**ไม่เสถียร** — ข้อมูลเปลี่ยนเล็กน้อยต้นไม้อาจเปลี่ยนรูปทั้งต้น ซึ่งแก้ได้ด้วย Ensemble ในหัวข้อถัดไป

## 6. Ensemble Learning และ Random Forest

### 6.1 Wisdom of the crowd

ถ้าถามคำถามยาก ๆ กับคนหลายพันคนแล้วรวมคำตอบ มักได้คำตอบที่ดีกว่าการถามผู้เชี่ยวชาญคนเดียว แนวคิดเดียวกันนี้ใช้กับโมเดลได้: การรวมคำทำนายจากหลายโมเดล (**ensemble**) มักดีกว่าโมเดลเดี่ยวที่ดีที่สุด — โดยเฉพาะเมื่อโมเดลย่อย**ผิดพลาดต่างกัน** (หลากหลาย ไม่ผิดซ้ำที่เดียวกัน)

| เทคนิค | หลักการ | ตัวอย่างใน scikit-learn |
|---|---|---|
| **Voting** | ใช้โมเดล*ต่างชนิด*กันมาโหวต — **hard voting** นับเสียงคลาสที่ทาย / **soft voting** เฉลี่ย `predict_proba` แล้วเลือกคลาสที่สูงสุด (ทุกโมเดลต้องมี `predict_proba`) | `VotingClassifier` |
| **Bagging** | ใช้โมเดล*ชนิดเดียวกัน*หลายตัว แต่ละตัวฝึกด้วยชุดข้อมูลที่สุ่มมา**แบบใส่คืน** (bootstrap) แล้วโหวต — ลด variance | `BaggingClassifier(bootstrap=True)` |
| **Pasting** | เหมือน bagging แต่สุ่ม**แบบไม่ใส่คืน** | `BaggingClassifier(bootstrap=False)` |
| **Random Forest** | Bagging ของ Decision Tree + สุ่มเลือกเซตย่อยของฟีเจอร์ทุกครั้งที่แบ่ง node → ต้นไม้หลากหลายยิ่งขึ้น | `RandomForestClassifier` |
| **Boosting** | ฝึกโมเดล*ต่อกันเป็นลำดับ* ตัวถัดไปพยายามแก้ข้อผิดพลาดของตัวก่อนหน้า — **AdaBoost** เพิ่มน้ำหนักให้ตัวอย่างที่ทายผิด / **Gradient Boosting** ให้โมเดลใหม่ทำนาย residual error ของโมเดลเดิม | `AdaBoostClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier` |

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (VotingClassifier, BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

log_clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
knn_clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)

models = {
    "Decision Tree (เดี่ยว)": tree_clf,
    "Voting (hard)": VotingClassifier(
        [("lr", log_clf), ("knn", knn_clf), ("dt", tree_clf)], voting="hard"),
    "Voting (soft)": VotingClassifier(
        [("lr", log_clf), ("knn", knn_clf), ("dt", tree_clf)], voting="soft"),
    "Bagging (500 trees)": BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500, bootstrap=True, n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:<24} test accuracy = {model.score(X_test, y_test):.3f}")
```

```output
Decision Tree (เดี่ยว)   test accuracy = 0.939
Voting (hard)            test accuracy = 0.982
Voting (soft)            test accuracy = 0.956
Bagging (500 trees)      test accuracy = 0.947
Random Forest            test accuracy = 0.956
AdaBoost                 test accuracy = 0.965
Gradient Boosting        test accuracy = 0.956
```

**การตีความ:** ทุกวิธีแบบ ensemble ดีกว่าต้นไม้เดี่ยว (0.939) ในชุดข้อมูลนี้ hard voting ได้ดีที่สุด ส่วน soft voting ต่ำกว่าเล็กน้อย — ไม่ได้ขัดกับทฤษฎี เพราะต้นไม้ที่ลึก 4 ชั้นให้ความน่าจะเป็นแบบ "มั่นใจเกินจริง" (เช่น 1.0) ซึ่งดึงค่าเฉลี่ยได้มาก soft voting จะได้เปรียบเมื่อทุกโมเดลให้ความน่าจะเป็นที่ calibrate ดี อีกทั้ง test set มีเพียง 114 ตัวอย่าง ความต่าง 0.01–0.03 คือต่างกันแค่ 1–3 ตัวอย่าง

### 6.2 Feature importance จาก Random Forest

Random Forest วัดได้ว่าแต่ละฟีเจอร์ช่วยลดความไม่บริสุทธิ์ (Gini) ได้มากแค่ไหนโดยเฉลี่ยในทุกต้น ผลรวมของทุกฟีเจอร์เท่ากับ 1 — มีประโยชน์มากสำหรับการทำความเข้าใจข้อมูลและการเลือกฟีเจอร์

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)

for name, score in zip(iris.data.columns, rnd_clf.feature_importances_):
    print(f"{name:<20} {score:.3f}")
```

```output
sepal length (cm)    0.112
sepal width (cm)     0.023
petal length (cm)    0.441
petal width (cm)     0.423
```

ขนาดกลีบดอก (petal) สำคัญกว่าขนาดกลีบเลี้ยง (sepal) มาก สอดคล้องกับที่ Decision Tree ในหัวข้อ 5 เลือกใช้แต่ petal length และ petal width

## 7. การประเมินผลโมเดล Classification

### 7.1 Confusion Matrix

**Confusion matrix** คือตารางนับว่าโมเดลทายถูกและผิดแบบไหนบ้าง สำหรับปัญหา binary มี 4 ช่อง ใน scikit-learn **แถว = คลาสจริง (actual), คอลัมน์ = คลาสที่ทาย (predicted)** เรียงคลาส 0 ก่อน 1

| | ทาย 0 (Negative) | ทาย 1 (Positive) |
|---|---|---|
| **จริง 0 (Negative)** | **TN** — True Negative | **FP** — False Positive (Type I error) |
| **จริง 1 (Positive)** | **FN** — False Negative (Type II error) | **TP** — True Positive |

วิธีจำ: คำแรก (True/False) บอกว่า**ทายถูกหรือไม่** คำหลัง (Positive/Negative) บอกว่า**โมเดลทายว่าอะไร** เช่น FN = ทายว่า negative แต่ทายผิด (จริง ๆ เป็น positive) — ในงานคัดกรองโรค FN คือ "ผู้ป่วยที่ถูกบอกว่าปกติ" ซึ่งอันตรายที่สุด

> [!WARNING]
> ตำราและเว็บไซต์บางแห่งวาง confusion matrix สลับกัน (เอา positive ไว้ก่อน หรือเอา predicted ไว้ที่แถว) ก่อนอ่านค่าทุกครั้งให้ดูหัวตารางก่อน สำหรับ scikit-learn ใช้ `tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()` ได้เสมอ

### 7.2 สูตร Metrics หลัก

| Metric | สูตร | ตอบคำถาม |
|---|---|---|
| **Accuracy** | $\dfrac{TP + TN}{TP + TN + FP + FN}$ | ทายถูกทั้งหมดกี่เปอร์เซ็นต์ |
| **Precision** | $\dfrac{TP}{TP + FP}$ | ในบรรดาที่**ทายว่า positive** ถูกจริงกี่เปอร์เซ็นต์ (ความน่าเชื่อถือของการแจ้งเตือน) |
| **Recall** (Sensitivity, TPR) | $\dfrac{TP}{TP + FN}$ | ในบรรดา**positive จริง**ทั้งหมด หาเจอกี่เปอร์เซ็นต์ (ความครอบคลุม) |
| **Specificity** (TNR) | $\dfrac{TN}{TN + FP}$ | ในบรรดา negative จริง ทายถูกว่า negative กี่เปอร์เซ็นต์ |
| **F1-score** | $2\cdot\dfrac{\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}} = \dfrac{2TP}{2TP + FP + FN}$ | ค่าเฉลี่ยฮาร์มอนิกของ precision กับ recall |

F1 ใช้ค่าเฉลี่ย**ฮาร์มอนิก** ไม่ใช่ค่าเฉลี่ยเลขคณิต เพราะต้องการลงโทษกรณีที่ค่าใดค่าหนึ่งต่ำมาก เช่น precision = 1.0, recall = 0.1 ค่าเฉลี่ยธรรมดาได้ 0.55 แต่ F1 ได้เพียง $2(1.0)(0.1)/(1.1) = 0.18$ — F1 จะสูงได้ก็ต่อเมื่อ**ทั้งสองค่า**สูง

### 7.3 ตัวอย่างการคำนวณด้วยมือ

โมเดลคัดกรองโรคชนิดหนึ่งถูกทดสอบกับคน 200 คน (ป่วยจริง 50 คน ไม่ป่วย 150 คน) ได้ผลดังนี้

| | ทาย "ไม่ป่วย" | ทาย "ป่วย" | รวม |
|---|---|---|---|
| **ไม่ป่วยจริง** | TN = 135 | FP = 15 | 150 |
| **ป่วยจริง** | FN = 10 | TP = 40 | 50 |
| **รวม** | 145 | 55 | 200 |

- Accuracy $= \dfrac{40 + 135}{200} = \dfrac{175}{200} = 0.875$
- Precision $= \dfrac{40}{40 + 15} = \dfrac{40}{55} = 0.727$ — ในคนที่โมเดลบอกว่าป่วย 55 คน ป่วยจริง 72.7%
- Recall $= \dfrac{40}{40 + 10} = \dfrac{40}{50} = 0.800$ — ผู้ป่วยจริง 50 คน โมเดลหาเจอ 80% (หลุดไป 10 คน)
- Specificity $= \dfrac{135}{135 + 15} = 0.900$
- F1 $= \dfrac{2(0.727)(0.800)}{0.727 + 0.800} = 0.762$ หรือ $\dfrac{2(40)}{2(40) + 15 + 10} = \dfrac{80}{105} = 0.762$

ตรวจคำตอบด้วย scikit-learn

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

# สร้างข้อมูลให้ได้ TN=135, FP=15, FN=10, TP=40 ตามตัวอย่าง
y_true = np.array([0]*135 + [0]*15 + [1]*10 + [1]*40)
y_pred = np.array([0]*135 + [1]*15 + [0]*10 + [1]*40)

cm = confusion_matrix(y_true, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()           # ลำดับของ sklearn: TN, FP, FN, TP
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

print(f"Accuracy  = {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision = {precision_score(y_true, y_pred):.4f}")
print(f"Recall    = {recall_score(y_true, y_pred):.4f}")
print(f"F1-score  = {f1_score(y_true, y_pred):.4f}")
print()
print(classification_report(y_true, y_pred, target_names=["healthy (0)", "disease (1)"], digits=3))
```

```output
[[135  15]
 [ 10  40]]
TN=135 FP=15 FN=10 TP=40
Accuracy  = 0.8750
Precision = 0.7273
Recall    = 0.8000
F1-score  = 0.7619

              precision    recall  f1-score   support

 healthy (0)      0.931     0.900     0.915       150
 disease (1)      0.727     0.800     0.762        50

    accuracy                          0.875       200
   macro avg      0.829     0.850     0.839       200
weighted avg      0.880     0.875     0.877       200
```

### 7.4 อ่าน `classification_report` และ macro vs weighted average

`classification_report` คำนวณ precision/recall/F1 **แยกทุกคลาส** โดยมองแต่ละคลาสเป็น positive ทีละคลาส เช่น แถว `healthy (0)` มองคลาส 0 เป็น positive: precision = 135/145 = 0.931, recall = 135/150 = 0.900 (ซึ่งก็คือ specificity ของคลาส 1) ส่วน `support` คือจำนวนตัวอย่างจริงของคลาสนั้น

- **macro avg** — ค่าเฉลี่ยธรรมดาของทุกคลาส ทุกคลาสสำคัญเท่ากัน: precision = (0.931 + 0.727)/2 = 0.829 → เหมาะเมื่อสนใจคลาสส่วนน้อยเท่ากับคลาสส่วนใหญ่
- **weighted avg** — เฉลี่ยถ่วงน้ำหนักด้วย support: precision = (0.931×150 + 0.727×50)/200 = 0.880 → ถูกคลาสส่วนใหญ่ครอบงำ ค่ามักใกล้ accuracy

เมื่อข้อมูลไม่สมดุล ถ้า macro avg ต่ำกว่า weighted avg มาก แปลว่าโมเดลทำงานกับคลาสส่วนน้อยได้แย่

### 7.5 Accuracy หลอกเราได้เมื่อข้อมูลไม่สมดุล

ลองสร้างโจทย์ binary จาก MNIST: "ภาพนี้คือเลข 5 หรือไม่" มีเพียงราว 9% ของภาพที่เป็นเลข 5 เปรียบเทียบโมเดลจริงกับ `DummyClassifier` ที่**ทายว่า "ไม่ใช่ 5" ทุกภาพ**โดยไม่ดูข้อมูลเลย (ใช้ชุดย่อย 10,000/2,000 ภาพเพื่อให้รันเร็ว)

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# ใช้ชุดย่อยเพื่อให้รันเร็ว: train 10,000 ภาพแรก / test 2,000 ภาพแรกของชุดทดสอบมาตรฐาน
X_train, y_train = X[:10_000], y[:10_000]
X_test,  y_test  = X[60_000:62_000], y[60_000:62_000]

# โจทย์ binary: "ภาพนี้คือเลข 5 หรือไม่"
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
print(f"เลข 5 ใน test set มี {y_test_5.sum()} จาก {len(y_test_5)} ภาพ ({y_test_5.mean():.2%})")

dummy = DummyClassifier(strategy="most_frequent")   # ทายว่า "ไม่ใช่ 5" ทุกภาพ
dummy.fit(X_train, y_train_5)
y_pred_dummy = dummy.predict(X_test)

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train_5)
y_pred_sgd = sgd.predict(X_test)

for name, y_pred in [("Dummy (always not-5)", y_pred_dummy), ("SGDClassifier", y_pred_sgd)]:
    print(f"{name:<22} acc={accuracy_score(y_test_5, y_pred):.3f}  "
          f"precision={precision_score(y_test_5, y_pred, zero_division=0):.3f}  "
          f"recall={recall_score(y_test_5, y_pred):.3f}  "
          f"f1={f1_score(y_test_5, y_pred):.3f}")
```

```output
เลข 5 ใน test set มี 179 จาก 2000 ภาพ (8.95%)
Dummy (always not-5)   acc=0.910  precision=0.000  recall=0.000  f1=0.000
SGDClassifier          acc=0.941  precision=0.624  recall=0.855  f1=0.722
```

**การตีความ:** โมเดลที่ "ไม่ได้เรียนรู้อะไรเลย" ได้ accuracy ถึง **91%** เพียงเพราะคลาส negative มีมาก แต่ recall = 0 คือหาเลข 5 ไม่เจอแม้แต่ภาพเดียว ส่วน SGDClassifier ได้ accuracy สูงกว่าแค่ 3% แต่ recall ต่างกันลิบลับ บทเรียนคือ **เมื่อข้อมูลไม่สมดุล (imbalanced) อย่าดูแค่ accuracy** ต้องดู confusion matrix, precision, recall และ F1 ของคลาสที่เราสนใจเสมอ และควรเทียบกับ baseline แบบ `DummyClassifier` ทุกครั้ง

### 7.6 Precision–Recall trade-off และการปรับ Threshold

Precision กับ recall มักสวนทางกัน: ถ้า**ลด threshold** โมเดลจะทายว่า positive ง่ายขึ้น → จับ positive ได้มากขึ้น (recall ↑) แต่ก็ทายผิดเป็น positive มากขึ้น (precision ↓) ถ้า**เพิ่ม threshold** จะเกิดผลตรงข้าม

| สถานการณ์ | ความผิดพลาดที่แพงกว่า | ให้ความสำคัญกับ |
|---|---|---|
| คัดกรองมะเร็ง / เบาหวาน | FN (ผู้ป่วยหลุดไม่ได้รักษา) | **Recall** สูง (ยอม FP บ้าง แล้วส่งตรวจซ้ำ) |
| ตรวจจับการโกงบัตรเครดิต | FN (ความเสียหายทางการเงิน) | **Recall** สูง |
| กรอง spam | FP (อีเมลสำคัญหายไปใน spam) | **Precision** สูง |
| แนะนำวิดีโอที่ปลอดภัยสำหรับเด็ก | FP (วิดีโอไม่เหมาะสมหลุดไปถึงเด็ก) | **Precision** สูง |

ตัวอย่างด้วยข้อมูลมะเร็งเต้านม โดยกำหนดให้ **malignant (เนื้อร้าย) = 1 เป็น positive** แล้วลองเปลี่ยน threshold

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score

data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)        # ให้ 1 = malignant (เนื้อร้าย) เป็น positive class
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]    # ความน่าจะเป็นของคลาส 1 (malignant)
for t in [0.9, 0.7, 0.5, 0.3, 0.1]:
    y_pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"threshold={t:.1f}  TP={tp:>2} FP={fp:>2} FN={fn:>2} TN={tn:>2}  "
          f"precision={precision_score(y_test, y_pred):.3f}  recall={recall_score(y_test, y_pred):.3f}")

print(f"\nROC AUC = {roc_auc_score(y_test, proba):.4f}")
```

```output
threshold=0.9  TP=33 FP= 0 FN= 9 TN=72  precision=1.000  recall=0.786
threshold=0.7  TP=38 FP= 0 FN= 4 TN=72  precision=1.000  recall=0.905
threshold=0.5  TP=39 FP= 1 FN= 3 TN=71  precision=0.975  recall=0.929
threshold=0.3  TP=41 FP= 1 FN= 1 TN=71  precision=0.976  recall=0.976
threshold=0.1  TP=41 FP= 5 FN= 1 TN=67  precision=0.891  recall=0.976
```

```output
ROC AUC = 0.9960
```

**การตีความ:** ที่ threshold 0.9 โมเดลระวังมาก — ทุกครั้งที่บอกว่าเป็นเนื้อร้ายถูกหมด (precision = 1.0) แต่ปล่อยผู้ป่วยหลุดไป 9 ราย เมื่อลด threshold ลงเหลือ 0.3 จับผู้ป่วยได้ 41 จาก 42 ราย โดย FP เพิ่มเพียง 1 ราย สำหรับงานแพทย์ threshold 0.3 จึงน่าสนใจกว่าค่าเริ่มต้น 0.5 แต่ถ้าลดถึง 0.1 recall ไม่เพิ่มอีกแล้ว มีแต่ FP ที่เพิ่ม

> [!WARNING]
> การเลือก threshold ควรทำบน **validation set** (หรือ cross-validation) ไม่ใช่บน test set เหมือนตัวอย่างนี้ ซึ่งทำเพื่อการสาธิตเท่านั้น ถ้าเลือก threshold จาก test set ผลบน test set จะดีเกินจริง

### 7.7 ROC curve และ AUC

**ROC curve** (Receiver Operating Characteristic) พล็อต **True Positive Rate** (= recall) บนแกน y เทียบกับ **False Positive Rate** $= \frac{FP}{FP + TN} = 1 - \text{specificity}$ บนแกน x โดยไล่ threshold ทุกค่า

- **AUC** (Area Under the Curve) คือพื้นที่ใต้กราฟ ROC มีค่า 0–1: AUC = 1.0 คือแยกสองคลาสได้สมบูรณ์ AUC = 0.5 เท่ากับการเดาสุ่ม (เส้นทแยง)
- ตีความได้ว่า: ถ้าสุ่มตัวอย่าง positive 1 ตัวและ negative 1 ตัว AUC คือโอกาสที่โมเดลให้คะแนน positive สูงกว่า negative — โมเดลข้างบนได้ 0.996 จึงแยกเนื้อร้ายกับเนื้อดีได้ดีมาก**โดยไม่ขึ้นกับ threshold**
- เมื่อคลาส positive หายากมาก ๆ (เช่น การโกง 0.1%) ควรดู **Precision–Recall curve** ประกอบ เพราะ ROC อาจดูดีเกินจริงเนื่องจาก TN จำนวนมหาศาลทำให้ FPR ต่ำเสมอ

```python
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[0], name="LogReg")
axes[0].set_title("Precision-Recall curve")
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name="LogReg")
axes[1].plot([0, 1], [0, 1], "k--", label="random (AUC=0.5)")
axes[1].legend()
axes[1].set_title("ROC curve")
plt.tight_layout()
plt.show()
```

เมื่อรันใน Colab จะได้กราฟ 2 รูป: PR curve ที่อยู่ชิดมุมขวาบน และ ROC curve ที่โค้งชิดมุมซ้ายบนห่างจากเส้นทแยงประ (การเดาสุ่ม) มาก

## 8. กรณีศึกษา: จำแนกตัวเลขลายมือ MNIST

MNIST (Modified National Institute of Standards and Technology) เปรียบเสมือน "Hello World" ของวงการ Machine Learning เป็นภาพตัวเลขเขียนด้วยลายมือ **70,000 ภาพ** (train 60,000 / test 10,000) แต่ละภาพขนาด **28 × 28 พิกเซล** เมื่อคลี่เป็น 1 มิติ (flatten) ได้ **784 features** แต่ละพิกเซลมีค่าความสว่าง 0 (ดำ) ถึง 255 (ขาว) และ label คือเลข 0–9 (**multiclass 10 คลาส**)

![ตัวอย่างภาพตัวเลขลายมือ 0–9 จากชุดข้อมูล MNIST](/images/m03/mnist_samples.png)

ลายมือของแต่ละคนต่างกันมาก บางคนเขียนเอียง บางคนเส้นหนา โมเดลต้องเรียนรู้ลักษณะเด่นที่แท้จริงของตัวเลขแต่ละตัว

![การกระจายตัวของจำนวนภาพในแต่ละคลาสของ MNIST ทั้ง 70,000 ภาพ](/images/m03/mnist_distribution.png)

ชุดข้อมูลค่อนข้าง**สมดุล** (ทุกคลาสมีราว 6,300–7,900 ภาพ) ต่างจากโจทย์ "เลข 5 หรือไม่" ในหัวข้อ 7.5 การใช้ **accuracy** เป็นตัวชี้วัดหลักจึงพอใช้ได้ แต่ยังควรดู confusion matrix เพื่อรู้ว่าโมเดลสับสนตรงไหน

### 8.1 เปรียบเทียบ 3 อัลกอริทึม

เพื่อให้รันได้เร็วในห้องเรียน ตัวอย่างนี้ใช้**ชุดย่อย train 10,000 ภาพ / test 2,000 ภาพ** (ถ้าใช้ชุดเต็ม accuracy จะสูงขึ้นอีก แต่ KNN จะ predict ช้ามาก)

```python
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)   # ปรับพิกเซล 0–255 ให้อยู่ในช่วง 0–1

# ชุดย่อย: train 10,000 / test 2,000 (ชุดเต็มคือ 60,000 / 10,000)
X_train, y_train = X[:10_000], y[:10_000]
X_test,  y_test  = X[60_000:62_000], y[60_000:62_000]
print("X_train:", X_train.shape, " X_test:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "KNN (k=3, distance)": KNeighborsClassifier(n_neighbors=3, weights="distance"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}
preds = {}
for name, model in models.items():
    t0 = time.perf_counter(); model.fit(X_train, y_train); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); preds[name] = model.predict(X_test); t_pred = time.perf_counter() - t0
    print(f"{name:<20} acc={accuracy_score(y_test, preds[name]):.4f}  "
          f"fit={t_fit:5.1f}s  predict={t_pred:5.2f}s")
```

```output
X_train: (10000, 784)  X_test: (2000, 784)
Logistic Regression  acc=0.8880  fit= 37.3s  predict= 0.00s
KNN (k=3, distance)  acc=0.9285  fit=  0.1s  predict= 2.63s
Random Forest        acc=0.9345  fit=  2.1s  predict= 0.08s
```

(เวลาที่ได้ขึ้นกับเครื่อง ใน Colab อาจต่างจากนี้ แต่สัดส่วนจะคล้ายกัน)

**การตีความ:**

- **Logistic Regression** ได้ 88.8% — เป็นโมเดลเชิงเส้น จึงจับลวดลายที่ซับซ้อนของลายมือได้จำกัด แต่ `predict` เร็วที่สุด (เป็นแค่การคูณเมทริกซ์) สังเกตว่า `fit` ใช้เวลานานที่สุดเพราะต้องหาค่าน้ำหนัก 785 × 10 ค่าด้วยการ optimize ซ้ำหลายรอบ
- **KNN** ได้ 92.9% โดย `fit` แทบไม่ใช้เวลาเลย (แค่จำข้อมูล) แต่ `predict` ช้าที่สุด เพราะทุกภาพทดสอบต้องวัดระยะกับภาพฝึกทั้ง 10,000 ภาพ ถ้าใช้ชุดเต็ม 60,000 ภาพจะช้าขึ้นอีกราว 6 เท่า (accuracy บนชุดเต็มจะได้ประมาณ 97%)
- **Random Forest** ได้สูงสุด 93.5% ทั้ง fit และ predict เร็ว — เป็นตัวเลือกเริ่มต้นที่ดีมาก

ข้อสังเกต: เราหาร 255 เพื่อปรับพิกเซลให้อยู่ช่วง 0–1 ซึ่งช่วยให้ Logistic Regression ลู่เข้าได้ง่ายขึ้น แต่ไม่จำเป็นสำหรับ Random Forest

### 8.2 วิเคราะห์ Confusion Matrix แบบหลายคลาส

สำหรับ 10 คลาส confusion matrix มีขนาด 10 × 10 — แถว $i$ คอลัมน์ $j$ คือจำนวนภาพที่เป็นเลข $i$ จริงแต่ถูกทายเป็นเลข $j$ ตัวเลขบน**เส้นทแยงมุม**คือทายถูก ตัวเลข**นอกเส้นทแยง**คือความผิดพลาด

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = preds["KNN (k=3, distance)"]
cm = confusion_matrix(y_test, y_pred)
print(cm)

# หาคู่ที่สับสนกันมากที่สุด (นอกเส้นทแยง)
off = cm.copy()
np.fill_diagonal(off, 0)
pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10)), reverse=True)[:6]
for count, i, j in pairs:
    print(f"เลขจริง {i} -> ทายเป็น {j}: {count} ภาพ")

print(classification_report(y_test, y_pred, digits=3))
```

```output
[[175   0   0   0   0   0   0   0   0   0]
 [  0 234   0   0   0   0   0   0   0   0]
 [  4   4 198   0   0   0   3   8   2   0]
 [  0   1   0 190   0   4   2   4   3   3]
 [  0   3   0   0 203   0   2   0   1   8]
 [  1   0   0   7   0 165   1   2   0   3]
 [  2   1   0   0   1   1 173   0   0   0]
 [  0  11   0   1   1   1   0 186   0   5]
 [  2   2   3   9   3   7   2   3 157   4]
 [  0   1   0   3   5   1   0   6   2 176]]
เลขจริง 7 -> ทายเป็น 1: 11 ภาพ
เลขจริง 8 -> ทายเป็น 3: 9 ภาพ
เลขจริง 4 -> ทายเป็น 9: 8 ภาพ
เลขจริง 2 -> ทายเป็น 7: 8 ภาพ
เลขจริง 8 -> ทายเป็น 5: 7 ภาพ
เลขจริง 5 -> ทายเป็น 3: 7 ภาพ
              precision    recall  f1-score   support

           0      0.951     1.000     0.975       175
           1      0.911     1.000     0.953       234
           2      0.985     0.904     0.943       219
           3      0.905     0.918     0.911       207
           4      0.953     0.935     0.944       217
           5      0.922     0.922     0.922       179
           6      0.945     0.972     0.958       178
           7      0.890     0.907     0.899       205
           8      0.952     0.818     0.880       192
           9      0.884     0.907     0.896       194

    accuracy                          0.928      2000
   macro avg      0.930     0.928     0.928      2000
weighted avg      0.930     0.928     0.928      2000
```

**การตีความ:**

- **คู่ที่หน้าตาคล้ายกันคือจุดอ่อน:** เลข 4 ↔ 9 สับสนกันรวม 13 ภาพ (4→9 จำนวน 8 ภาพ และ 9→4 จำนวน 5 ภาพ) เพราะเลข 4 ที่เขียนด้านบนปิดจะคล้ายเลข 9 มาก กลุ่ม **3 / 5 / 8** ก็สับสนกันบ่อย (8→3 = 9, 8→5 = 7, 5→3 = 7) เพราะต่างกันเพียงเส้นโค้งเล็ก ๆ ด้านซ้าย และเลข 7 ที่เขียนเป็นเส้นตรงถูกทายเป็นเลข 1 ถึง 11 ภาพ
- **เลข 8 ยากที่สุด** — recall = 0.818 (157/192) เพราะถูกทายไปเป็นเลขอื่นแทบทุกตัว แต่ precision สูง (0.952) แปลว่าเมื่อโมเดลกล้าทายว่า "8" ส่วนใหญ่ถูก
- **เลข 1 ตรงข้าม** — recall = 1.000 (หาเลข 1 เจอครบ) แต่ precision เพียง 0.911 เพราะเลขอื่น (โดยเฉพาะ 7) ถูกดึงมาทายเป็น 1 — KNN วัดระยะแบบพิกเซลต่อพิกเซล ภาพที่มีเส้นน้อย ๆ จึงใกล้กับเลข 1
- ข้อมูลค่อนข้างสมดุล macro avg กับ weighted avg จึงแทบเท่ากัน (0.928) และใกล้ accuracy

การรู้ว่าโมเดลผิดตรงไหนบอกทางแก้ได้ เช่น เพิ่มข้อมูลเลข 8 และ 4/9 ที่เขียนยาก ทำ data augmentation (หมุน/เลื่อนภาพเล็กน้อย) หรือเปลี่ยนไปใช้ CNN ที่เข้าใจรูปร่างของภาพ

วาด confusion matrix เป็นภาพ (heatmap) ใน Colab — แนะนำให้วาดแบบ normalize ด้วย เพื่อดู "อัตราการผิด" แทนจำนวน

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion matrix (counts)")

# normalize="true": หารด้วยจำนวนจริงของแต่ละแถว -> เห็น recall ของแต่ละคลาส
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1], cmap="Reds", colorbar=False,
                                        normalize="true", values_format=".2f")
axes[1].set_title("Normalized by row (recall)")
plt.tight_layout()
plt.show()
```

> [!NOTE]
> **ขั้นต่อไป: CNN** — โมเดลในบทนี้มองภาพเป็นเวกเตอร์ 784 ค่าโดยไม่รู้ว่าพิกเซลไหนอยู่ติดกัน จึงจับ "รูปร่าง" ได้จำกัด Convolutional Neural Network (CNN) ใช้ฟิลเตอร์กวาดภาพเพื่อตรวจจับเส้นขอบ มุม และลวดลาย ทำให้ได้ accuracy บน MNIST เกิน 99% เราจะสร้าง CNN ด้วย Keras ใน [บทที่ 7](/module/07)

## 9. การเลือกอัลกอริทึม Classification

ไม่มีอัลกอริทึมใดดีที่สุดสำหรับทุกปัญหา (No Free Lunch theorem) ตารางนี้สรุปแนวทางเลือกเบื้องต้น

| อัลกอริทึม | ต้อง scale? | อธิบายผลได้ | ความเร็ว fit / predict | เหมาะกับ | ข้อควรระวัง |
|---|---|---|---|---|---|
| Logistic Regression | ควร | ดี (ดูค่า `coef_`) | เร็ว / เร็วมาก | baseline แรก, ข้อมูลแยกได้เกือบเชิงเส้น, ต้องการความน่าจะเป็น | จับความสัมพันธ์ไม่เชิงเส้นไม่ได้ (ยกเว้นสร้างฟีเจอร์เพิ่ม) |
| KNN | **ต้อง** | ปานกลาง (ดูเพื่อนบ้าน) | ทันที / **ช้ามาก** | ข้อมูลน้อย–กลาง มิติไม่สูง | ช้าเมื่อข้อมูลเยอะ แย่ลงเมื่อมิติสูง |
| SVM (kernel) | **ต้อง** | ยาก | ช้าเมื่อเกินแสนแถว / ปานกลาง | ข้อมูลขนาดเล็ก–กลางที่มิติสูง เส้นแบ่งซับซ้อน | ต้องจูน `C`, `gamma` |
| Decision Tree | ไม่ต้อง | **ดีมาก** (white-box) | เร็ว / เร็ว | ต้องอธิบายกฎการตัดสินใจได้ ข้อมูลผสมหลายชนิด | overfit ง่าย ไม่เสถียร |
| Random Forest | ไม่ต้อง | ปานกลาง (`feature_importances_`) | ปานกลาง / ปานกลาง | ข้อมูลตาราง (tabular) ทั่วไป — ตัวเลือก "ใช้ได้ดีเกือบทุกครั้ง" | โมเดลใหญ่ ช้ากว่าต้นไม้เดี่ยว |
| Gradient Boosting | ไม่ต้อง | ปานกลาง | ช้า (ต่อกันเป็นลำดับ) / เร็ว | ต้องการ accuracy สูงสุดบนข้อมูลตาราง | ต้องจูนหลายพารามิเตอร์ overfit ได้ถ้าจำนวนรอบมากเกิน |
| CNN ([บทที่ 7](/module/07)) | normalize | ยาก | ช้า (ควรมี GPU) / ปานกลาง | ภาพ, ข้อมูลเชิงพื้นที่ | ต้องการข้อมูลมาก |

**ขั้นตอนแนะนำในทางปฏิบัติ:** (1) เริ่มจาก `DummyClassifier` และ Logistic Regression เป็น baseline (2) ลอง Random Forest / Gradient Boosting สำหรับข้อมูลตาราง (3) เปรียบเทียบด้วย cross-validation และ metric ที่ตรงกับเป้าหมายทางธุรกิจ (4) จูน hyperparameter และ threshold บน validation set (5) ประเมินบน test set **ครั้งเดียว**ตอนท้าย

## สรุป

- Classification ทำนายคำตอบที่เป็นหมวดหมู่ แบ่งเป็น binary, multiclass และ multilabel ตัวจำแนกใน scikit-learn ใช้ `fit` / `predict` / `predict_proba` เหมือนกันทุกตัว
- **Logistic Regression** = สมการเชิงเส้น + sigmoid ให้ความน่าจะเป็น ตัดสินด้วย threshold (ค่าเริ่มต้น 0.5) หลายคลาสใช้ softmax อัตโนมัติ (ไม่ต้องใช้ `multi_class` แล้ว)
- **KNN** โหวตจากเพื่อนบ้าน $k$ ตัว **SVM** หาถนนที่กว้างที่สุดและใช้ kernel trick — ทั้งคู่**ต้อง scale** ข้อมูล
- **Decision Tree** (CART) แบ่งข้อมูลด้วย Gini/entropy อ่านเข้าใจง่ายแต่ overfit ง่าย **Ensemble** (voting, bagging, Random Forest, boosting) รวมหลายโมเดลให้แม่นยำและเสถียรขึ้น
- **Confusion matrix** ของ scikit-learn: แถว = จริง, คอลัมน์ = ทาย, `ravel()` ได้ TN, FP, FN, TP
- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = ค่าเฉลี่ยฮาร์มอนิก — **accuracy หลอกได้เมื่อข้อมูลไม่สมดุล**
- ลด threshold → recall ↑ precision ↓ เลือกตามต้นทุนของ FP กับ FN และเลือกบน validation set; ROC AUC วัดความสามารถในการแยกคลาสโดยไม่ขึ้นกับ threshold

## คำศัพท์สำคัญ

| ศัพท์ | ความหมาย |
|---|---|
| Decision boundary | เส้น/พื้นผิวที่แบ่งเขตระหว่างคลาสใน feature space |
| Sigmoid / Logistic function | $\sigma(z) = 1/(1+e^{-z})$ แปลงค่าใด ๆ ให้อยู่ในช่วง 0–1 |
| Softmax | ฟังก์ชันแปลงคะแนนของ $K$ คลาสเป็นความน่าจะเป็นที่รวมกันได้ 1 |
| Decision threshold | ค่าความน่าจะเป็นขั้นต่ำที่จะตัดสินว่าเป็น positive |
| Support vectors | ข้อมูลบนขอบ "ถนน" ของ SVM ซึ่งกำหนดตำแหน่งเส้นแบ่ง |
| Kernel trick | เทคนิคที่ให้ผลเหมือนแปลงข้อมูลไปมิติสูงโดยไม่ต้องคำนวณฟีเจอร์ใหม่จริง |
| Gini impurity | $1-\sum p_k^2$ วัดความไม่บริสุทธิ์ของ node ใน Decision Tree |
| Entropy | $-\sum p_k\log_2 p_k$ วัดความไม่แน่นอนของ node |
| Bagging / Pasting | ฝึกโมเดลหลายตัวบนข้อมูลที่สุ่มแบบใส่คืน / ไม่ใส่คืน แล้วรวมผล |
| Boosting | ฝึกโมเดลต่อกันเป็นลำดับ ตัวใหม่แก้ข้อผิดพลาดของตัวก่อน |
| Confusion matrix | ตารางนับผลการทายเทียบกับคลาสจริง (TN, FP, FN, TP) |
| Precision | สัดส่วนที่ทายว่า positive แล้วถูก = TP/(TP+FP) |
| Recall (Sensitivity) | สัดส่วน positive จริงที่หาเจอ = TP/(TP+FN) |
| F1-score | ค่าเฉลี่ยฮาร์มอนิกของ precision และ recall |
| Imbalanced data | ข้อมูลที่จำนวนตัวอย่างแต่ละคลาสต่างกันมาก |
| ROC AUC | พื้นที่ใต้กราฟ TPR–FPR วัดความสามารถในการแยกคลาสโดยไม่ขึ้นกับ threshold |

## คำถามทบทวนท้ายบท

**1.** โมเดลตรวจจับธุรกรรมทุจริตได้ผลบน test set ดังนี้: TN = 940, FP = 20, FN = 15, TP = 25 จงคำนวณ accuracy, precision, recall และ F1-score

<details><summary>เฉลย</summary>

รวม = 940 + 20 + 15 + 25 = 1,000

- Accuracy = (25 + 940)/1000 = **0.965**
- Precision = 25/(25 + 20) = 25/45 = **0.556**
- Recall = 25/(25 + 15) = 25/40 = **0.625**
- F1 = 2(25)/(2(25) + 20 + 15) = 50/85 = **0.588**

สังเกตว่า accuracy สูงถึง 96.5% แต่โมเดลจับการทุจริตได้เพียง 62.5% เพราะธุรกรรมทุจริตมีเพียง 4% ของข้อมูล

</details>

**2.** ทำไมโมเดลที่ทายว่า "ไม่ใช่เลข 5" ทุกภาพจึงได้ accuracy ประมาณ 91% และควรใช้ metric ใดแทน

<details><summary>เฉลย</summary>

เพราะภาพเลข 5 มีเพียงประมาณ 9% ของข้อมูล การทายว่า negative ทุกภาพจึงถูก 91% โดยอัตโนมัติ ทั้งที่ไม่ได้เรียนรู้อะไรเลย ควรดู **recall** (ซึ่งเป็น 0 — หาเลข 5 ไม่เจอเลย) precision และ F1-score ของคลาส positive รวมถึง confusion matrix และเปรียบเทียบกับ `DummyClassifier` เป็น baseline

</details>

**3.** โรงพยาบาลต้องการโมเดลคัดกรองมะเร็งระยะเริ่มต้น ควรปรับ threshold ของ Logistic Regression ให้สูงขึ้นหรือต่ำลงจาก 0.5 เพราะเหตุใด และต้องแลกกับอะไร

<details><summary>เฉลย</summary>

ควร**ลด** threshold (เช่น 0.3) เพื่อให้โมเดลทายว่า "เป็นมะเร็ง" ง่ายขึ้น → recall สูงขึ้น ผู้ป่วยหลุด (FN) น้อยลง ซึ่งเป็นความผิดพลาดที่ร้ายแรงที่สุดในงานนี้ สิ่งที่ต้องแลกคือ precision ลดลง (FP เพิ่ม) คนที่ไม่ป่วยบางส่วนต้องไปตรวจยืนยันเพิ่ม ซึ่งมีต้นทุนต่ำกว่าการพลาดผู้ป่วยจริง ทั้งนี้ควรเลือก threshold จาก validation set

</details>

**4.** node หนึ่งของ Decision Tree มีตัวอย่าง 3 คลาสจำนวน [30, 10, 0] จงคำนวณ Gini impurity และบอกว่าถ้าเปลี่ยนเป็น [40, 0, 0] ค่า Gini จะเป็นเท่าใด

<details><summary>เฉลย</summary>

สัดส่วน = 30/40 = 0.75, 10/40 = 0.25, 0
Gini = 1 − (0.75² + 0.25² + 0²) = 1 − (0.5625 + 0.0625) = **0.375**

ถ้าเป็น [40, 0, 0] → Gini = 1 − 1² = **0** (บริสุทธิ์สมบูรณ์)

</details>

**5.** เพราะเหตุใด KNN และ SVM จึงต้อง scale ข้อมูล แต่ Decision Tree และ Random Forest ไม่จำเป็น

<details><summary>เฉลย</summary>

KNN และ SVM ใช้ **ระยะทาง** ระหว่างจุด (KNN วัดระยะไปเพื่อนบ้าน SVM วัดความกว้างของ margin) ฟีเจอร์ที่มีหน่วยใหญ่จึงครอบงำระยะทาง ส่วน Decision Tree แบ่ง node โดยเปรียบเทียบ**ฟีเจอร์ทีละตัว**กับค่าเกณฑ์ (เช่น `glucose <= 127.5`) การเปลี่ยนสเกลแบบเชิงเส้นแค่ย้ายค่าเกณฑ์ แต่ลำดับของข้อมูลไม่เปลี่ยน ผลการแบ่งจึงเหมือนเดิม

</details>

**6.** `classification_report` ของโมเดลหนึ่งแสดง macro avg F1 = 0.62 แต่ weighted avg F1 = 0.91 บอกอะไรเกี่ยวกับโมเดลนี้

<details><summary>เฉลย</summary>

ข้อมูลไม่สมดุล และโมเดลทำงานได้ดีกับ**คลาสส่วนใหญ่** (ซึ่งมีน้ำหนักมากใน weighted avg) แต่ทำงานได้แย่กับ**คลาสส่วนน้อย** (macro avg ให้น้ำหนักทุกคลาสเท่ากันจึงถูกดึงลง) ควรดู F1 ของแต่ละคลาสแยก แล้วพิจารณาวิธีช่วยคลาสส่วนน้อย เช่น `class_weight="balanced"` หรือปรับ threshold

</details>

## อ่านเพิ่มเติม

- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.) — Chapter 3 Classification (MNIST, confusion matrix, precision/recall, ROC), Chapter 4 (Logistic & Softmax Regression), Chapter 5 SVM, Chapter 6 Decision Trees, Chapter 7 Ensemble Learning and Random Forests
- scikit-learn User Guide: [Model evaluation — classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- scikit-learn User Guide: [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html), [Decision Trees](https://scikit-learn.org/stable/modules/tree.html), [Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
- scikit-learn: [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html) (`TunedThresholdClassifierCV`)
- ใบงานประจำบท: [Lab 03 — Classification และการคำนวณ Metrics](/lab/03)
