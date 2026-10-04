## ข้อมูลทั่วไปของใบงาน

| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Classification และการคำนวณ Metrics |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 4 |
| งานตาม มคอ.3 | แบบฝึกหัด 3: classification และ metrics |
| CLO/LLO | CLO2 (LLO2-2 เลือกอัลกอริทึม ML ให้เหมาะสมกับปัญหา, LLO2-3 พัฒนาแบบจำลองด้วย Python/scikit-learn และประเมินผลด้วย accuracy, precision, recall, F1-score) |
| เครื่องมือ | Google Colab, Python 3, pandas, NumPy, matplotlib, scikit-learn |
| ข้อมูล | `pima_indians_diabetes_real.csv` (768 แถว 8 features + `Outcome`) และ `load_digits` จาก scikit-learn |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ + กระดาษคำนวณด้วยมือข้อที่ 1 (ถ่ายรูปแนบใน notebook) + คำตอบคำถามท้ายข้อ |

### วัตถุประสงค์

1. คำนวณ accuracy, precision, recall, specificity และ F1-score จาก confusion matrix ด้วยมือ แล้วตรวจสอบด้วย scikit-learn ได้
2. ฝึกและเปรียบเทียบตัวจำแนก 5 แบบ (Logistic Regression, KNN, SVM, Decision Tree, Random Forest) บนข้อมูลเดียวกันอย่างเป็นธรรมได้
3. อ่าน confusion matrix และ `classification_report` แล้วตีความในบริบทของงานคัดกรองโรคได้
4. ปรับ decision threshold เพื่อเพิ่ม recall โดยเลือก threshold จาก cross-validation (ไม่ใช่จาก test set) ได้
5. วิเคราะห์ความผิดพลาดของโมเดลแบบหลายคลาส (ตัวเลขลายมือ) จาก confusion matrix ขนาด 10 × 10 ได้

> [!NOTE]
> ใบงานนี้ต่อยอดจาก [บทที่ 3](/module/03) และใช้ข้อมูลเบาหวานชุดเดียวกับ [Lab 02](/lab/02) ให้รันเซลล์ตามลำดับจากบนลงล่าง ทุกเซลล์ที่มี `# TODO:` รันได้ตั้งแต่ต้น (ด้วยค่าชั่วคราว `None`) แต่ผลลัพธ์จะครบและถูกต้องเมื่อเติมโค้ดแล้วเท่านั้น

## ข้อที่ 0: เตรียมสภาพแวดล้อม

ดาวน์โหลดไฟล์ `pima_indians_diabetes_real.csv` จากหน้ารายวิชา แล้วรันเซลล์ด้านล่างเพื่ออัปโหลดเข้า Colab

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)

RANDOM_STATE = 42
pd.set_option("display.width", 120)

try:
    from google.colab import files
    files.upload()  # เลือกไฟล์ pima_indians_diabetes_real.csv
except ImportError:
    print("ไม่ได้รันบน Colab — วางไฟล์ CSV ไว้ในโฟลเดอร์เดียวกับ notebook")

df = pd.read_csv("pima_indians_diabetes_real.csv")
print(df.shape)
```

ผลลัพธ์ที่ควรได้คือ `(768, 9)`

## ข้อที่ 1: คำนวณ Metrics จาก Confusion Matrix ด้วยมือ

### โจทย์

ชุดตรวจคัดกรองเบาหวานแบบเร็วชนิดหนึ่งถูกทดสอบกับอาสาสมัคร 150 คน (ป่วยจริง 50 คน ไม่ป่วย 100 คน) ได้ confusion matrix ดังนี้ (positive = ป่วย, จัดวางแบบ scikit-learn: แถว = จริง, คอลัมน์ = ทาย)

| | ทาย "ไม่ป่วย" (0) | ทาย "ป่วย" (1) |
|---|---|---|
| **ไม่ป่วยจริง (0)** | TN = 88 | FP = 12 |
| **ป่วยจริง (1)** | FN = 18 | TP = 32 |

1. **คำนวณด้วยมือบนกระดาษก่อน** (แสดงสูตรและการแทนค่า ทศนิยม 4 ตำแหน่ง): accuracy, precision, recall, specificity และ F1-score
2. เติมสูตรในโค้ดให้คำนวณค่าเดียวกัน
3. รันเซลล์ตรวจสอบด้วย scikit-learn แล้วเทียบว่าตรงกับที่คำนวณด้วยมือหรือไม่

### โค้ดเริ่มต้น

```python
TN, FP, FN, TP = 88, 12, 18, 32

accuracy    = None  # TODO: สัดส่วนที่ทายถูกทั้งหมด
precision   = None  # TODO: ในบรรดาที่ทายว่าป่วย ป่วยจริงกี่ส่วน
recall      = None  # TODO: ในบรรดาผู้ป่วยจริง หาเจอกี่ส่วน
f1          = None  # TODO: ค่าเฉลี่ยฮาร์มอนิกของ precision และ recall
specificity = None  # TODO: ในบรรดาคนไม่ป่วยจริง ทายถูกว่าไม่ป่วยกี่ส่วน

print("Accuracy   :", accuracy)
print("Precision  :", precision)
print("Recall     :", recall)
print("F1-score   :", f1)
print("Specificity:", specificity)
```

เซลล์ตรวจสอบ (ไม่ต้องแก้) — สร้าง `y_true` และ `y_pred` ที่ให้ confusion matrix เดียวกับโจทย์ แล้วให้ scikit-learn คำนวณ

```python
y_true_1 = np.array([0] * (TN + FP) + [1] * (FN + TP))
y_pred_1 = np.array([0] * TN + [1] * FP + [0] * FN + [1] * TP)

print(confusion_matrix(y_true_1, y_pred_1))
print(f"sklearn -> acc={accuracy_score(y_true_1, y_pred_1):.4f}  "
      f"prec={precision_score(y_true_1, y_pred_1):.4f}  "
      f"rec={recall_score(y_true_1, y_pred_1):.4f}  "
      f"f1={f1_score(y_true_1, y_pred_1):.4f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
Accuracy   : 0.8
Precision  : 0.7272727272727273
Recall     : 0.64
F1-score   : 0.6808510638297872
Specificity: 0.88
[[88 12]
 [18 32]]
sklearn -> acc=0.8000  prec=0.7273  rec=0.6400  f1=0.6809
```

### คำถามท้ายข้อ

1. ชุดตรวจนี้มี accuracy 80% แต่ recall เพียง 64% หมายความว่าอย่างไรในทางปฏิบัติ (ผู้ป่วยกี่คนจาก 50 คนที่ถูกบอกว่า "ปกติ")
2. ถ้าโรงพยาบาลใช้ชุดตรวจนี้คัดกรอง ระหว่าง FP 12 คน กับ FN 18 คน กลุ่มใดมีต้นทุน/ความเสี่ยงสูงกว่า เพราะเหตุใด
3. ถ้าชุดตรวจ "ทายว่าไม่ป่วยทุกคน" จะได้ accuracy, recall และ precision เท่าใด (คำนวณด้วยมือ)

## ข้อที่ 2: เตรียมข้อมูลเบาหวาน (ทบทวน Lab 02)

### โจทย์

รันเซลล์ด้านล่างซึ่งทำขั้นตอนเดียวกับ Lab 02: แปลงเลข 0 ที่เป็นค่าว่างแฝงให้เป็น `NaN` แล้วแบ่ง train/test แบบ stratified 80/20 (การเติมค่าว่างจะทำใน `Pipeline` ในข้อถัดไปเพื่อป้องกัน data leakage)

### โค้ดเริ่มต้น

```python
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)

print(df["Outcome"].value_counts())
print(df.isna().sum()[zero_as_missing])

X = df.drop(columns="Outcome")
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print("train:", X_train.shape, " test:", X_test.shape)
print(f"สัดส่วนผู้ป่วยใน test = {y_test.mean():.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
Outcome
0    500
1    268
Name: count, dtype: int64
Glucose            5
BloodPressure     35
SkinThickness    227
Insulin          374
BMI               11
dtype: int64
train: (614, 8)  test: (154, 8)
สัดส่วนผู้ป่วยใน test = 0.351
```

### คำถามท้ายข้อ

1. ข้อมูลนี้ balanced หรือ imbalanced ถ้าโมเดลทายว่า "ไม่ป่วย" ทุกคนจะได้ accuracy บน test set ประมาณเท่าใด
2. ทำไมต้องใช้ `stratify=y` ในการแบ่งข้อมูล

## ข้อที่ 3: ฝึกและเปรียบเทียบตัวจำแนก 5 แบบ

### โจทย์

1. ฟังก์ชัน `make_model(clf)` ห่อตัวจำแนกด้วย `SimpleImputer(median)` → `StandardScaler` → ตัวจำแนก (ตัวอย่าง Logistic Regression ทำให้ดูแล้ว)
2. เติมโมเดลที่เหลือ: `KNeighborsClassifier(n_neighbors=15)`, `SVC(kernel="rbf", C=1)`, `DecisionTreeClassifier(max_depth=4)` และ `RandomForestClassifier(n_estimators=300)` (ใส่ `random_state=RANDOM_STATE` ให้โมเดลที่มีความสุ่ม)
3. รันลูปเพื่อคำนวณ accuracy, precision, recall และ F1 บน test set แล้วเรียงตาม F1
4. แสดง `feature_importances_` ของ Random Forest

### โค้ดเริ่มต้น

```python
def make_model(clf):
    """ห่อตัวจำแนกด้วยการเติมค่าว่างและ scaling เพื่อให้ทุกโมเดลได้ข้อมูลแบบเดียวกัน"""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)

models = {
    "Logistic Regression": make_model(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "KNN (k=15)":          None,  # TODO: make_model(KNeighborsClassifier(...))
    "SVM (RBF)":           None,  # TODO: make_model(SVC(...))
    "Decision Tree":       None,  # TODO: make_model(DecisionTreeClassifier(...))
    "Random Forest":       None,  # TODO: make_model(RandomForestClassifier(...))
}

results = []
for name, model in models.items():
    if model is None:
        print(f"ข้าม {name} (ยังไม่ได้สร้างโมเดล)")
        continue
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).set_index("model").round(3)
print(results_df.sort_values("f1", ascending=False))
```

```python
rf_pipe = models["Random Forest"]
if rf_pipe is not None:
    rf = rf_pipe[-1]                      # ขั้นสุดท้ายของ pipeline คือ RandomForestClassifier
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.round(3))
```

### ผลลัพธ์ที่คาดหวัง

```output
                     accuracy  precision  recall     f1
model                                                  
Decision Tree           0.786      0.698   0.685  0.692
SVM (RBF)               0.740      0.652   0.556  0.600
KNN (k=15)              0.740      0.652   0.556  0.600
Random Forest           0.740      0.659   0.537  0.592
Logistic Regression     0.708      0.600   0.500  0.545
```

```output
Glucose                     0.268
BMI                         0.160
DiabetesPedigreeFunction    0.124
Age                         0.121
Insulin                     0.091
BloodPressure               0.085
Pregnancies                 0.079
SkinThickness               0.072
dtype: float64
```

### คำถามท้ายข้อ

1. โมเดลใดได้ F1 สูงสุดบน test set และ recall ของทุกโมเดลอยู่ในช่วงใด ค่า recall ระดับนี้ยอมรับได้หรือไม่สำหรับงานคัดกรองเบาหวาน
2. test set มีผู้ป่วยเพียง 54 คน ความต่างของ recall 0.019 เท่ากับทายถูกต่างกันกี่คน เราควรสรุปว่า "Decision Tree ดีที่สุด" ได้ทันทีหรือไม่ ควรใช้วิธีใดยืนยัน (คำใบ้: cross-validation)
3. ฟีเจอร์ใดสำคัญที่สุด 3 อันดับแรกตาม Random Forest สอดคล้องกับความรู้ทางการแพทย์หรือไม่
4. Decision Tree และ Random Forest จำเป็นต้องใช้ `StandardScaler` หรือไม่ ทำไมเรายังใส่ไว้ใน `make_model` ได้โดยไม่เสียหาย

## ข้อที่ 4: Confusion Matrix และ Classification Report

### โจทย์

1. ใช้ Logistic Regression จากข้อที่ 3 ทำนาย test set แล้วสร้าง confusion matrix
2. แยกค่า TN, FP, FN, TP ด้วย `.ravel()` แล้ว**คำนวณ recall เอง**จาก TP และ FN
3. พิมพ์ `classification_report` และวาด confusion matrix ด้วย `ConfusionMatrixDisplay`

### โค้ดเริ่มต้น

```python
log_model = models["Logistic Regression"]
y_pred_lr = log_model.predict(X_test)

cm = None  # TODO: confusion_matrix(y_test, y_pred_lr)

if cm is not None:
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
    recall_manual = None  # TODO: คำนวณ recall จาก tp และ fn
    print("recall (คำนวณเอง) =", recall_manual)

    print(classification_report(y_test, y_pred_lr, target_names=["No diabetes", "Diabetes"], digits=3))
    ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot(cmap="Blues")
    plt.title("Logistic Regression - Pima test set")
    plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
[[82 18]
 [27 27]]
TN=82 FP=18 FN=27 TP=27
recall (คำนวณเอง) = 0.5
              precision    recall  f1-score   support

 No diabetes      0.752     0.820     0.785       100
    Diabetes      0.600     0.500     0.545        54

    accuracy                          0.708       154
   macro avg      0.676     0.660     0.665       154
weighted avg      0.699     0.708     0.701       154
```

และได้ภาพ heatmap ขนาด 2 × 2 ที่ช่องมุมซ้ายบน (TN) เข้มที่สุด

### คำถามท้ายข้อ

1. FN = 27 หมายถึงอะไรในบริบทของงานนี้ อธิบายเป็นภาษาที่แพทย์เข้าใจ
2. ตรวจสอบด้วยมือว่า precision ของแถว `No diabetes` (0.752) คำนวณมาจากช่องใดของ confusion matrix
3. คำนวณ macro avg ของ recall ด้วยมือจากค่าในตาราง และอธิบายว่าทำไม weighted avg จึงใกล้ accuracy มากกว่า macro avg

## ข้อที่ 5: ปรับ Threshold เพื่อเพิ่ม Recall

### โจทย์

ทีมแพทย์ต้องการให้โมเดลจับผู้ป่วยได้ **อย่างน้อย 80%** (recall ≥ 0.80) โดยยอมให้มีผลบวกลวงมากขึ้นได้

1. ใช้ `cross_val_predict(..., method="predict_proba")` หาความน่าจะเป็นของ **train set** แบบ 5-fold (เราไม่เลือก threshold จาก test set เพราะจะทำให้ผลบน test ดีเกินจริง)
2. รันตารางเปรียบเทียบ precision/recall ที่ threshold 0.1–0.9
3. **เลือก threshold** ที่ให้ recall ≥ 0.80 และ precision สูงที่สุด จาก `thr_df`
4. นำ threshold ที่เลือกไปใช้กับ test set แล้วเทียบกับ threshold 0.5

### โค้ดเริ่มต้น

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
proba_cv = cross_val_predict(log_model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

rows = []
for t in np.arange(0.1, 0.91, 0.1):
    y_pred_t = (proba_cv >= t).astype(int)
    rows.append({"threshold": round(t, 1),
                 "precision": precision_score(y_train, y_pred_t, zero_division=0),
                 "recall": recall_score(y_train, y_pred_t),
                 "f1": f1_score(y_train, y_pred_t)})
thr_df = pd.DataFrame(rows).round(3)
print(thr_df.to_string(index=False))
```

```python
best_t = None  # TODO: เลือกจาก thr_df: กรองแถวที่ recall >= 0.80 แล้วเอา threshold ของแถวที่ precision สูงสุด

if best_t is not None:
    print("threshold ที่เลือก =", best_t)
    proba_test = log_model.predict_proba(X_test)[:, 1]
    for t in [0.5, best_t]:
        y_pred_t = (proba_test >= t).astype(int)
        print(f"\nthreshold = {t}")
        print(confusion_matrix(y_test, y_pred_t))
        print(f"precision={precision_score(y_test, y_pred_t):.3f}  recall={recall_score(y_test, y_pred_t):.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
 threshold  precision  recall    f1
       0.1      0.445   0.991 0.614
       0.2      0.519   0.893 0.656
       0.3      0.588   0.780 0.671
       0.4      0.689   0.682 0.685
       0.5      0.759   0.575 0.654
       0.6      0.766   0.491 0.598
       0.7      0.840   0.416 0.556
       0.8      0.853   0.271 0.411
       0.9      0.821   0.107 0.190
```

```output
threshold ที่เลือก = 0.2

threshold = 0.5
[[82 18]
 [27 27]]
precision=0.600  recall=0.500

threshold = 0.2
[[61 39]
 [ 6 48]]
precision=0.552  recall=0.889
```

### คำถามท้ายข้อ

1. เมื่อเปลี่ยน threshold จาก 0.5 เป็น 0.2 จำนวน FN และ FP บน test set เปลี่ยนจากเท่าใดเป็นเท่าใด คุ้มค่าหรือไม่สำหรับงานคัดกรอง
2. ทำไมการเลือก threshold จาก test set โดยตรงจึงถือว่า "โกง" (data leakage)
3. ถ้าเป้าหมายเปลี่ยนเป็น "ส่งต่อพบแพทย์เฉพาะคนที่โมเดลมั่นใจมาก เพราะคิวแพทย์มีจำกัด" ควรขยับ threshold ไปทางใด และ metric ใดจะสำคัญที่สุด

## ข้อที่ 6: Multiclass — จำแนกตัวเลขลายมือ (load_digits)

### โจทย์

`load_digits` เป็นภาพตัวเลขลายมือ 0–9 ขนาด 8 × 8 พิกเซล (64 features) จำนวน 1,797 ภาพ — เป็น MNIST ฉบับย่อที่ฝึกได้ในไม่กี่วินาที

1. แบ่งข้อมูล train/test 70/30 แบบ stratified (ทำให้แล้ว)
2. สร้าง `digit_model` = `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))` แล้ว fit
3. สร้าง confusion matrix 10 × 10 และหา 5 คู่ (เลขจริง → เลขที่ทาย) ที่สับสนกันมากที่สุด
4. พิมพ์ `classification_report` แล้ววิเคราะห์ว่าเลขใดยากที่สุด

### โค้ดเริ่มต้น

```python
from sklearn.datasets import load_digits

digits = load_digits()
Xd, yd = digits.data, digits.target
print(Xd.shape, np.bincount(yd))

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, stratify=yd, random_state=RANDOM_STATE)

digit_model = None  # TODO: make_pipeline(StandardScaler(), LogisticRegression(...)) แล้ว .fit(Xd_train, yd_train)

if digit_model is not None:
    yd_pred = digit_model.predict(Xd_test)
    print(f"accuracy = {accuracy_score(yd_test, yd_pred):.4f}")
```

```python
if digit_model is not None:
    cm_d = confusion_matrix(yd_test, yd_pred)
    print(cm_d)

    off = cm_d.copy()
    np.fill_diagonal(off, 0)              # ลบเส้นทแยง (ทายถูก) ออก เหลือแต่ความผิดพลาด
    pairs = [(off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j] > 0]
    for count, i, j in sorted(pairs, reverse=True)[:5]:
        print(f"เลขจริง {i} ถูกทายเป็น {j}: {count} ครั้ง")

    print(classification_report(yd_test, yd_pred, digits=3))
    ConfusionMatrixDisplay(cm_d).plot(cmap="Blues", colorbar=False)
    plt.title("Digits - Logistic Regression")
    plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
(1797, 64) [178 182 177 183 181 182 181 179 174 180]
accuracy = 0.9815
```

```output
[[54  0  0  0  0  0  0  0  0  0]
 [ 0 52  0  0  1  0  0  0  2  0]
 [ 0  0 53  0  0  0  0  0  0  0]
 [ 0  0  0 55  0  0  0  0  0  0]
 [ 0  0  0  0 53  0  0  0  1  0]
 [ 0  0  0  0  0 54  0  0  0  1]
 [ 0  0  0  0  0  0 53  0  1  0]
 [ 0  0  0  0  0  0  0 54  0  0]
 [ 0  3  0  0  0  0  0  0 49  0]
 [ 0  0  0  0  0  0  0  0  1 53]]
เลขจริง 8 ถูกทายเป็น 1: 3 ครั้ง
เลขจริง 1 ถูกทายเป็น 8: 2 ครั้ง
เลขจริง 9 ถูกทายเป็น 8: 1 ครั้ง
เลขจริง 6 ถูกทายเป็น 8: 1 ครั้ง
เลขจริง 5 ถูกทายเป็น 9: 1 ครั้ง
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        54
           1      0.945     0.945     0.945        55
           2      1.000     1.000     1.000        53
           3      1.000     1.000     1.000        55
           4      0.981     0.981     0.981        54
           5      1.000     0.982     0.991        55
           6      1.000     0.981     0.991        54
           7      1.000     1.000     1.000        54
           8      0.907     0.942     0.925        52
           9      0.981     0.981     0.981        54

    accuracy                          0.981       540
   macro avg      0.982     0.981     0.981       540
weighted avg      0.982     0.981     0.982       540
```

### คำถามท้ายข้อ

1. คำนวณด้วยมือจาก confusion matrix: precision และ recall ของเลข 8 (ดูคอลัมน์ 8 และแถว 8) ให้ตรงกับรายงาน
2. ทำไมข้อมูลชุดนี้จึงใช้ accuracy เป็นตัวชี้วัดหลักได้ ต่างจากโจทย์เบาหวานอย่างไร
3. คู่ที่สับสนมากที่สุดคือคู่ใด ลองใช้ `plt.imshow(Xd_test[k].reshape(8, 8), cmap="gray")` ดูภาพที่ทายผิดสัก 2 ภาพ แล้วอธิบายว่าทำไมโมเดลจึงสับสน
4. (ท้าทาย ไม่บังคับ) เปลี่ยน `digit_model` เป็น `KNeighborsClassifier(n_neighbors=3)` หรือ `SVC()` แล้วเปรียบเทียบ accuracy และคู่ที่สับสน — ถ้ามีเวลา ลองโค้ด MNIST ชุดย่อยในหัวข้อ 8 ของบทเรียน แล้วตรวจว่าคู่ 4 ↔ 9 และ 3 / 5 / 8 สับสนกันจริงหรือไม่

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | รายละเอียดในใบงานนี้ | คะแนน |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | อธิบายความหมายของ TP/FP/FN/TN, precision, recall, F1, macro/weighted avg และผลของ threshold ได้ถูกต้อง | 2 |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | เลือก metric ที่เหมาะกับงานคัดกรองโรค (recall) เลือกโมเดลและ threshold อย่างมีเหตุผล ไม่ตัดสินจากความต่างเล็กน้อยบน test set | 2 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | เตรียมข้อมูล → Pipeline → ฝึก 5 โมเดล → ประเมิน → เลือก threshold ด้วย CV → ทดสอบ → วิเคราะห์ multiclass | 2 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | คำนวณ metrics ด้วยมือถูกต้อง (ข้อ 1, 4, 6) เติม TODO ครบ รันได้ ผลตรงกับที่คาดหวัง | 3 |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | ตีความ FN/FP ในบริบทจริง และระบุข้อจำกัด (test set เล็ก, ข้อมูลจากประชากรเฉพาะกลุ่ม) | 1 |
| **รวม** | | **10** |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab03_รหัสนักศึกษา.ipynb` ที่เติม TODO ครบ และรันทุกเซลล์จนเห็นผลลัพธ์ (Runtime → Run all)
2. รูปถ่ายกระดาษคำนวณด้วยมือของข้อที่ 1 (แสดงสูตร การแทนค่า และคำตอบ) แทรกในเซลล์ Markdown ใต้ข้อที่ 1
3. คำตอบคำถามท้ายข้อทุกข้อ เขียนในเซลล์ Markdown ใต้ข้อนั้น ๆ
4. สรุปท้าย notebook 5–8 บรรทัด: โมเดลและ threshold ที่คุณจะแนะนำให้ใช้คัดกรองเบาหวาน พร้อมเหตุผลจากตัวเลข และข้อจำกัดของผลการทดลอง
5. ส่งภายในวันก่อนเริ่มคาบเรียนสัปดาห์ที่ 5 ผ่านช่องทางที่ผู้สอนกำหนด
