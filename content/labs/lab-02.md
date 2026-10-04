## ข้อมูลทั่วไปของใบงาน

| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อใบงาน | เตรียมข้อมูลและ Train/Test Split (Diabetes) |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 3 |
| งานตาม มคอ.3 | งานปฏิบัติ 1: เตรียมข้อมูลและ train/test split |
| CLO/LLO | CLO2 (LLO2-1 เตรียมข้อมูล เลือกคุณลักษณะ แบ่งข้อมูล และปรับขนาดข้อมูล, LLO2-2 เลือกอัลกอริทึมให้เหมาะกับปัญหา) |
| เครื่องมือ | Google Colab, Python 3, pandas, NumPy, matplotlib, scikit-learn |
| ข้อมูล | `pima_indians_diabetes_real.csv` (768 แถว 8 features + `Outcome`) |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ พร้อมคำตอบคำถามท้ายข้อ |

### วัตถุประสงค์

1. สำรวจข้อมูลด้วย pandas และตรวจพบค่าว่างแฝง (เลข 0 ที่แทนค่าที่ไม่ได้วัด) ได้
2. เลือกวิธีจัดการค่าว่างและ outlier ได้อย่างมีเหตุผล
3. แบ่งข้อมูล train/test แบบ stratified และเลือก feature จาก train set ได้
4. ปรับสเกลข้อมูลโดยไม่เกิด data leakage และรวมทุกขั้นไว้ใน `Pipeline` ได้
5. สร้างโมเดล baseline (Logistic Regression) ประเมินด้วย accuracy และ recall และทำนายคนไข้รายใหม่ได้

> [!NOTE]
> ใบงานนี้ต่อยอดจากเนื้อหา [บทที่ 2](/module/02) ให้รันเซลล์ตามลำดับจากบนลงล่าง ทุกเซลล์ที่มี `# TODO:` รันได้ตั้งแต่ต้น (ด้วยค่าชั่วคราว) แต่ผลลัพธ์จะถูกต้องเมื่อเติมโค้ดครบแล้วเท่านั้น

## ข้อที่ 0: เตรียมสภาพแวดล้อม

ดาวน์โหลดไฟล์ `pima_indians_diabetes_real.csv` จากหน้ารายวิชา แล้วรันเซลล์ด้านล่างเพื่ออัปโหลดเข้า Colab

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

try:
    from google.colab import files
    files.upload()  # เลือกไฟล์ pima_indians_diabetes_real.csv
except ImportError:
    print("ไม่ได้รันบน Colab — วางไฟล์ CSV ไว้ในโฟลเดอร์เดียวกับ notebook")

df = pd.read_csv("pima_indians_diabetes_real.csv")
print(df.shape)
```

ผลลัพธ์ที่ควรได้คือ `(768, 9)`

## ข้อที่ 1: โหลดและสำรวจข้อมูลเบื้องต้น

### โจทย์

1. แสดง 5 แถวแรก และโครงสร้างข้อมูลด้วย `info()`
2. แสดงสถิติเชิงพรรณนาของทุกคอลัมน์ด้วย `describe()` (transpose และปัดทศนิยม 2 ตำแหน่ง)
3. หาสัดส่วนของแต่ละคลาสใน `Outcome` ด้วย `value_counts(normalize=True)`

### โค้ดเริ่มต้น

```python
print(df.head())
df.info()

# TODO: แสดงสถิติเชิงพรรณนา -> df.describe().T.round(2)
stats = df.head(0)
print(stats)

# TODO: สัดส่วนคลาสของ Outcome -> df["Outcome"].value_counts(normalize=True).round(3)
class_ratio = None
print(class_ratio)
```

### ผลลัพธ์ที่คาดหวัง

```output
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1
<class 'pandas.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
                          count    mean     std    min    25%     50%     75%     max
Pregnancies               768.0    3.85    3.37   0.00   1.00    3.00    6.00   17.00
Glucose                   768.0  120.89   31.97   0.00  99.00  117.00  140.25  199.00
BloodPressure             768.0   69.11   19.36   0.00  62.00   72.00   80.00  122.00
SkinThickness             768.0   20.54   15.95   0.00   0.00   23.00   32.00   99.00
Insulin                   768.0   79.80  115.24   0.00   0.00   30.50  127.25  846.00
BMI                       768.0   31.99    7.88   0.00  27.30   32.00   36.60   67.10
DiabetesPedigreeFunction  768.0    0.47    0.33   0.08   0.24    0.37    0.63    2.42
Age                       768.0   33.24   11.76  21.00  24.00   29.00   41.00   81.00
Outcome                   768.0    0.35    0.48   0.00   0.00    0.00    1.00    1.00
Outcome
0    0.651
1    0.349
Name: proportion, dtype: float64
```

### คำถามท้ายข้อ

1. คอลัมน์ใดบ้างที่มีค่า `min` เป็น 0 และคอลัมน์ใดที่ค่า 0 **เป็นไปไม่ได้** ทางการแพทย์
2. ข้อมูลนี้สมดุล (balanced) หรือไม่ ถ้าโมเดลทายว่า "ไม่เป็นเบาหวาน" ทุกคน จะได้ accuracy ประมาณเท่าใด

## ข้อที่ 2: ค้นหาค่าว่างแฝง (Disguised Missing Values)

### โจทย์

1. ตรวจว่า pandas พบค่า NaN กี่ค่า
2. นับจำนวนค่า 0 ในคอลัมน์ `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` พร้อมคิดเป็นร้อยละ
3. สร้าง `df_clean` โดยแทนค่า 0 ใน 5 คอลัมน์นี้ด้วย `np.nan` แล้วนับค่าว่างอีกครั้ง

### โค้ดเริ่มต้น

```python
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("NaN ที่ pandas เห็น:", df.isna().sum().sum())

# TODO: นับจำนวนค่า 0 ในแต่ละคอลัมน์ของ zero_cols -> (df[zero_cols] == 0).sum()
n_zero = pd.Series(0, index=zero_cols)
print(pd.DataFrame({"n_zero": n_zero, "percent": (n_zero / len(df) * 100).round(1)}))

df_clean = df.copy()
# TODO: แทนค่า 0 ใน zero_cols ด้วย np.nan -> df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)

print("ค่าว่างหลังแปลง:", df_clean.isna().sum().to_dict())
```

### ผลลัพธ์ที่คาดหวัง

```output
NaN ที่ pandas เห็น: 0
               n_zero  percent
Glucose             5      0.7
BloodPressure      35      4.6
SkinThickness     227     29.6
Insulin           374     48.7
BMI                11      1.4
ค่าว่างหลังแปลง: {'Pregnancies': 0, 'Glucose': 5, 'BloodPressure': 35, 'SkinThickness': 227, 'Insulin': 374, 'BMI': 11, 'DiabetesPedigreeFunction': 0, 'Age': 0, 'Outcome': 0}
```

### คำถามท้ายข้อ

1. ทำไม `df.isna().sum()` จึงตรวจไม่พบปัญหานี้ และเราต้องใช้ข้อมูลอะไรในการตัดสินว่าเลข 0 คือค่าว่าง
2. ทำไมเราไม่แปลงเลข 0 ในคอลัมน์ `Pregnancies` เป็นค่าว่างด้วย

## ข้อที่ 3: ลบทิ้ง หรือ เติมค่า?

### โจทย์

1. ลองลบทุกแถวที่มีค่าว่างด้วย `dropna()` แล้วดูว่าเหลือข้อมูลกี่แถว
2. เปรียบเทียบ mean และ median ของ `Insulin` (เฉพาะค่าที่ไม่ว่าง)
3. เติมค่าว่างใน `zero_cols` ด้วย `SimpleImputer(strategy="median")` (ในข้อนี้ fit กับข้อมูลทั้งหมดเพื่อ **สำรวจ** เท่านั้น — โมเดลจริงจะ fit กับ train set ในข้อที่ 8)

### โค้ดเริ่มต้น

```python
from sklearn.impute import SimpleImputer

# TODO: ลบแถวที่มีค่าว่าง -> df_clean.dropna()
df_drop = df_clean
print("ก่อนลบ:", df_clean.shape, " หลังลบ:", df_drop.shape)

ins = df_clean["Insulin"].dropna()
print(f"Insulin  mean = {ins.mean():.1f}   median = {ins.median():.1f}")

imputer = SimpleImputer(strategy="median")
# TODO: fit_transform คอลัมน์ zero_cols ของ df_clean -> imputer.fit_transform(df_clean[zero_cols])
filled = df_clean[zero_cols].to_numpy()
print("ค่าว่างที่เหลือหลังเติม:", int(np.isnan(filled).sum()))
```

### ผลลัพธ์ที่คาดหวัง

```output
ก่อนลบ: (768, 9)  หลังลบ: (392, 9)
Insulin  mean = 155.5   median = 125.0
ค่าว่างที่เหลือหลังเติม: 0
```

### คำถามท้ายข้อ

1. การลบแถวที่มีค่าว่างทำให้เสียข้อมูลไปกี่เปอร์เซ็นต์ และอาจทำให้เกิด bias อย่างไร
2. ทำไม median จึงเหมาะกว่า mean สำหรับเติม `Insulin` (ดูจากค่าที่พิมพ์ออกมา)

## ข้อที่ 4: ตรวจหา Outlier ด้วย IQR

### โจทย์

1. เขียนฟังก์ชัน `iqr_bounds(s)` คืนค่าขอบล่าง $Q_1 - 1.5 \times IQR$ และขอบบน $Q_3 + 1.5 \times IQR$
2. นับจำนวน outlier ของทุก feature (ใช้ `df_clean` และข้ามค่าว่าง)
3. วาด box plot ของ `BloodPressure`, `BMI` และ `Age`

### โค้ดเริ่มต้น

```python
def iqr_bounds(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = 0       # TODO: q3 - q1
    lower = q1    # TODO: q1 - 1.5 * iqr
    upper = q3    # TODO: q3 + 1.5 * iqr
    return lower, upper

rows = []
for col in df_clean.columns.drop("Outcome"):
    s = df_clean[col].dropna()
    lower, upper = iqr_bounds(s)
    n_out = int(((s < lower) | (s > upper)).sum())
    rows.append((col, round(lower, 2), round(upper, 2), n_out))
print(pd.DataFrame(rows, columns=["feature", "lower", "upper", "n_outliers"]).to_string(index=False))

df_clean[["BloodPressure", "BMI", "Age"]].plot(kind="box", subplots=True, figsize=(10, 3))
plt.tight_layout()
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
                 feature  lower  upper  n_outliers
             Pregnancies  -6.50  13.50           4
                 Glucose  36.00 204.00           0
           BloodPressure  40.00 104.00          14
           SkinThickness   1.00  57.00           3
                 Insulin -94.38 360.62          24
                     BMI  13.85  50.25           8
DiabetesPedigreeFunction  -0.33   1.20          29
                     Age  -1.50  66.50           9
```

### คำถามท้ายข้อ

1. คำนวณด้วยมือ: `Age` มี $Q_1 = 24$, $Q_3 = 41$ ขอบบนเท่ากับเท่าใด ตรงกับตารางหรือไม่
2. คนไข้อายุ 81 ปี หรือ `BloodPressure` 122 mmHg ควรลบทิ้งหรือไม่ เพราะอะไร เสนอทางเลือกอื่นแทนการลบ 1 วิธี

## ข้อที่ 5: แบ่งข้อมูล Train/Test แบบ Stratified

### โจทย์

1. แยก `X` (8 features จาก `df_clean`) และ `y` (`Outcome`)
2. แบ่ง train 80% / test 20% โดยรักษาสัดส่วนคลาสด้วย `stratify=y` และ `random_state=42`
3. แสดงขนาดและสัดส่วนคลาส 1 ของข้อมูลทั้งหมด train และ test

### โค้ดเริ่มต้น

```python
from sklearn.model_selection import train_test_split

X = df_clean.drop(columns="Outcome")
y = df_clean["Outcome"]

# TODO: เพิ่ม stratify=y เพื่อรักษาสัดส่วนคลาส
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, " Test:", X_test.shape)
print(f"สัดส่วนคลาส 1 — ทั้งหมด: {y.mean():.3f} | train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
Train: (614, 8)  Test: (154, 8)
สัดส่วนคลาส 1 — ทั้งหมด: 0.349 | train: 0.349 | test: 0.351
```

### คำถามท้ายข้อ

1. ทำไม test set มี 154 แถว (แสดงการคำนวณจาก 768 แถว)
2. ลองลบ `stratify=y` แล้วรันใหม่ สัดส่วนคลาส 1 ใน test เปลี่ยนเป็นเท่าใด เพราะเหตุใดการ stratify จึงสำคัญกับข้อมูลชุดนี้
3. ทำไมเราจึงแบ่งข้อมูล **ก่อน** เติมค่าว่าง เลือก feature และปรับสเกล

## ข้อที่ 6: Correlation และการเลือก Feature (บน Train Set)

### โจทย์

1. คำนวณ correlation ของแต่ละ feature กับ `Outcome` จาก **train set** เรียงจากมากไปน้อย แล้วเทียบกับ correlation ของ `Insulin` ในข้อมูลดิบ `df`
2. วาด heatmap ของ correlation matrix
3. ใช้ `SelectKBest(f_classif, k=4)` (ต่อจาก `SimpleImputer`) เลือก 4 features ที่ดีที่สุด

### โค้ดเริ่มต้น

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

train_df = X_train.assign(Outcome=y_train)
corr = train_df.corr()

# TODO: correlation กับ Outcome เรียงจากมากไปน้อย -> corr["Outcome"].drop("Outcome").sort_values(ascending=False)
corr_outcome = corr["Outcome"]
print(corr_outcome.round(3))
print("Insulin corr (ข้อมูลดิบที่มีเลข 0):", round(df["Insulin"].corr(df["Outcome"]), 3))

plt.figure(figsize=(7, 6))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson r")
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation matrix (train set)")
plt.tight_layout()
plt.show()

# TODO: เปลี่ยน k=8 เป็น k=4
selector = make_pipeline(SimpleImputer(strategy="median"), SelectKBest(f_classif, k=8))
selector.fit(X_train, y_train)
skb = selector.named_steps["selectkbest"]
print(pd.Series(skb.scores_, index=X_train.columns).sort_values(ascending=False).round(1))
print("features ที่เลือก:", list(X_train.columns[skb.get_support()]))
```

### ผลลัพธ์ที่คาดหวัง

```output
Glucose                     0.515
Insulin                     0.351
BMI                         0.331
SkinThickness               0.280
Age                         0.241
Pregnancies                 0.208
BloodPressure               0.188
DiabetesPedigreeFunction    0.165
Name: Outcome, dtype: float64
Insulin corr (ข้อมูลดิบที่มีเลข 0): 0.131
Glucose                     217.8
BMI                          74.2
Insulin                      39.4
Age                          37.6
SkinThickness                35.5
Pregnancies                  27.7
BloodPressure                21.3
DiabetesPedigreeFunction     17.2
dtype: float64
features ที่เลือก: ['Glucose', 'Insulin', 'BMI', 'Age']
```

### คำถามท้ายข้อ

1. correlation ของ `Insulin` กับ `Outcome` เปลี่ยนไปอย่างไรหลังแปลงเลข 0 เป็นค่าว่าง ข้อนี้สอนอะไรเกี่ยวกับลำดับขั้น "ทำความสะอาดก่อนเลือก feature"
2. ถ้าต้องสร้างแอปคัดกรองเบาหวานสำหรับ รพ.สต. ที่ไม่มีเครื่องตรวจอินซูลิน คุณจะเลือก features ชุดใด ใช้ทั้งผลจาก `SelectKBest` และความรู้เชิงโดเมนประกอบการตัดสินใจ

## ข้อที่ 7: ปรับสเกลข้อมูลโดยไม่เกิด Data Leakage

### โจทย์

1. เติมค่าว่างด้วย median ที่ fit จาก **train** แล้ว transform ทั้ง train และ test
2. ใช้ `StandardScaler` โดย `fit_transform` กับ train และ `transform` กับ test แสดง mean/std ของแต่ละชุด
3. เปรียบเทียบ $\mu$ ที่ได้จาก train เท่านั้น กับ $\mu$ ที่ได้จาก train+test (แบบที่ leak)
4. คำนวณ Min-Max scaling ของ `Glucose` (train) ด้วยสูตร $x' = (x - x_{min})/(x_{max} - x_{min})$ เอง แล้วเทียบกับ `MinMaxScaler`

### โค้ดเริ่มต้น

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

imp = SimpleImputer(strategy="median").fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)

scaler = StandardScaler()
# TODO: fit_transform กับ train -> scaler.fit_transform(X_train_imp)
X_train_s = X_train_imp
# TODO: transform กับ test (ห้าม fit!) -> scaler.transform(X_test_imp)
X_test_s = X_test_imp

print("train mean:", X_train_s.mean(axis=0).round(2), " std:", X_train_s.std(axis=0).round(2))
print("test  mean:", X_test_s.mean(axis=0).round(2), " std:", X_test_s.std(axis=0).round(2))

mu_train = StandardScaler().fit(X_train_imp).mean_
mu_leak = StandardScaler().fit(np.vstack([X_train_imp, X_test_imp])).mean_
print(pd.DataFrame({"mu_train_only": mu_train, "mu_train+test (leak)": mu_leak}, index=X.columns).round(2))

g = X_train_imp[:, X.columns.get_loc("Glucose")]
# TODO: Min-Max ด้วยสูตร -> (g - g.min()) / (g.max() - g.min())
g_manual = g
g_sklearn = MinMaxScaler().fit_transform(g.reshape(-1, 1)).ravel()
print("min/max ของ g_manual:", g_manual.min(), g_manual.max(), " ตรงกับ MinMaxScaler:", np.allclose(g_manual, g_sklearn))
```

### ผลลัพธ์ที่คาดหวัง

```output
train mean: [-0. -0.  0. -0. -0.  0. -0. -0.]  std: [1. 1. 1. 1. 1. 1. 1. 1.]
test  mean: [ 0.04 -0.    0.1   0.04  0.19  0.01 -0.08 -0.05]  std: [1.08 1.07 0.92 0.94 1.41 1.04 1.01 0.97]
                          mu_train_only  mu_train+test (leak)
Pregnancies                        3.82                  3.85
Glucose                          121.67                121.66
BloodPressure                     72.14                 72.39
SkinThickness                     29.04                 29.11
Insulin                          137.71                140.67
BMI                               32.45                 32.46
DiabetesPedigreeFunction           0.48                  0.47
Age                               33.37                 33.24
min/max ของ g_manual: 0.0 1.0  ตรงกับ MinMaxScaler: True
```

### คำถามท้ายข้อ

1. ทำไม mean ของ test หลัง scale จึง **ไม่เท่ากับ 0 พอดี** และนี่เป็นสิ่งที่ผิดหรือถูก
2. อธิบายด้วยคำพูดของตัวเองว่า data leakage คืออะไร และในตาราง $\mu$ ด้านบน ข้อมูลของ test รั่วเข้าไปที่ใด
3. ถ้าเปลี่ยนโมเดลเป็น Random Forest จำเป็นต้อง scale หรือไม่ ถ้าเป็น KNN ล่ะ เพราะอะไร

## ข้อที่ 8: Pipeline, โมเดล Baseline และการทำนายคนไข้รายใหม่

### โจทย์

1. สร้าง `ColumnTransformer`: คอลัมน์ใน `zero_cols` ให้ impute (median) แล้ว scale ส่วน `Pregnancies`, `DiabetesPedigreeFunction`, `Age` ให้ scale อย่างเดียว
2. ต่อเข้ากับ `LogisticRegression(max_iter=1000)` ใน `Pipeline` แล้ว fit กับ train set
3. ประเมินบน test set ด้วย accuracy, recall และ confusion matrix เทียบกับ `DummyClassifier`
4. ทำนายคนไข้รายใหม่: อายุ 35 ปี ตั้งครรภ์ 2 ครั้ง น้ำตาล 110 ความดัน 80 BMI 25.0 DPF 0.35 **ไม่ได้ตรวจ** SkinThickness และ Insulin

### โค้ดเริ่มต้น

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

other_cols = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]

preprocess = ColumnTransformer([
    ("missing", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), zero_cols),
    ("complete", StandardScaler(), other_cols),
])

# TODO: เปลี่ยน DummyClassifier เป็น LogisticRegression(max_iter=1000)
clf = Pipeline([
    ("prep", preprocess),
    ("model", DummyClassifier(strategy="most_frequent")),
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

print(f"Model  accuracy = {accuracy_score(y_test, y_pred):.3f}  recall = {recall_score(y_test, y_pred):.3f}")
print(f"Dummy  accuracy = {accuracy_score(y_test, y_dummy):.3f}  recall = {recall_score(y_test, y_dummy):.3f}")
print("Confusion matrix (แถว = ค่าจริง 0/1, คอลัมน์ = ค่าทำนาย 0/1):")
print(confusion_matrix(y_test, y_pred))

new_patient = pd.DataFrame([{
    "Pregnancies": 2, "Glucose": 110, "BloodPressure": 80, "SkinThickness": np.nan,
    "Insulin": np.nan, "BMI": 25.0, "DiabetesPedigreeFunction": 0.35, "Age": 35,
}])
# TODO: ทำนายคลาสและความน่าจะเป็นของคลาส 1 -> clf.predict(new_patient), clf.predict_proba(new_patient)[:, 1]
pred_class = None
pred_proba = None
print("ผลทำนาย:", pred_class, " ความน่าจะเป็นเป็นเบาหวาน:", pred_proba)
```

### ผลลัพธ์ที่คาดหวัง

```output
Model  accuracy = 0.708  recall = 0.500
Dummy  accuracy = 0.649  recall = 0.000
Confusion matrix (แถว = ค่าจริง 0/1, คอลัมน์ = ค่าทำนาย 0/1):
[[82 18]
 [27 27]]
ผลทำนาย: [0]  ความน่าจะเป็นเป็นเบาหวาน: [0.085]
```

### ท้าทาย (ไม่บังคับ): เพิ่ม Recall

ในงานคัดกรองโรค การพลาดผู้ป่วยจริง (False Negative) อันตรายกว่าการเตือนผิด ลองเปรียบเทียบโมเดลปกติกับ `class_weight="balanced"` ด้วย 5-fold cross-validation **บน train set** (ไม่ใช้ test set เลือกโมเดล)

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

for cw in [None, "balanced"]:
    pipe = Pipeline([("prep", clone(preprocess)),
                     ("model", LogisticRegression(max_iter=1000, class_weight=cw))])
    rec = cross_val_score(pipe, X_train, y_train, cv=5, scoring="recall")
    acc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"class_weight={cw!s:9}  CV recall = {rec.mean():.3f}  CV accuracy = {acc.mean():.3f}")
```

```output
class_weight=None       CV recall = 0.584  CV accuracy = 0.782
class_weight=balanced   CV recall = 0.701  CV accuracy = 0.752
```

### คำถามท้ายข้อ

1. โมเดล Dummy ได้ accuracy เท่าใด และ recall เท่าใด ทำไม accuracy ของ Dummy จึงไม่ต่ำอย่างที่คิด
2. จาก confusion matrix ของ Logistic Regression มีผู้ป่วยจริงกี่คนที่โมเดลทายว่า "ปกติ" (False Negative) คำนวณ recall ด้วยมือให้ตรงกับที่ scikit-learn รายงาน
3. คนไข้รายใหม่ไม่ได้ตรวจ Insulin และ SkinThickness แต่โมเดลยังทำนายได้ เพราะขั้นตอนใดใน Pipeline และค่าที่ใช้เติมมาจากข้อมูลชุดใด
4. ข้อดีของการใช้ `Pipeline` เทียบกับการเรียก `fit`/`transform` ทีละขั้นเอง (ข้อที่ 7) มีอะไรบ้าง อย่างน้อย 2 ข้อ
5. (ท้าทาย) `class_weight="balanced"` เปลี่ยน recall และ accuracy อย่างไร คุณจะเลือกโมเดลใดสำหรับงานคัดกรองเบาหวาน เพราะเหตุใด

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | รายละเอียดในใบงานนี้ | คะแนน |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | อธิบายค่าว่างแฝง, IQR, stratified split, data leakage และเหตุผลของการใช้ recall ได้ถูกต้อง | 2 |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | เลือก drop/impute, จัดการ outlier, เลือก features และ scaler ได้อย่างมีเหตุผล ระบุได้ว่าเป็นงาน binary classification | 2 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | ทำครบตั้งแต่สำรวจ → ทำความสะอาด → แบ่งข้อมูล → เลือก feature → scale → Pipeline → ประเมิน → ทำนาย | 2 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | เติม TODO ครบ รันได้ ผลลัพธ์ตรงกับที่คาดหวัง คำนวณ IQR และ recall ด้วยมือถูกต้อง | 3 |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | ตีความ accuracy vs recall และระบุข้อจำกัดของโมเดล (เช่น ข้อมูลจากประชากรเฉพาะกลุ่ม) | 1 |
| **รวม** | | **10** |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab02_รหัสนักศึกษา.ipynb` ที่เติม TODO ครบ และรันทุกเซลล์จนเห็นผลลัพธ์ (Runtime → Run all)
2. คำตอบคำถามท้ายข้อทุกข้อ เขียนในเซลล์ Markdown ใต้ข้อนั้น ๆ
3. สรุปท้าย notebook 5–8 บรรทัด: ขั้นตอนการเตรียมข้อมูลที่คุณคิดว่าสำคัญที่สุด 2 ขั้น พร้อมเหตุผล และข้อจำกัดของโมเดลที่ได้
4. ส่งภายในวันก่อนเริ่มคาบเรียนสัปดาห์ที่ 4 ผ่านช่องทางที่ผู้สอนกำหนด
