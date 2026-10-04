## ข้อมูลทั่วไปของใบงาน

| รายการ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 04: Regression ด้วย scikit-learn (California Housing) |
| งานตาม มคอ.3 | งานปฏิบัติ 2: regression ด้วย scikit-learn |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 5 |
| CLO/LLO | CLO2 — LLO2-2 (เลือกอัลกอริทึมให้เหมาะสม), LLO2-3 (พัฒนาแบบจำลองด้วย scikit-learn และประเมินผลด้วย MSE ฯลฯ) |
| เครื่องมือ | Google Colab, Python 3, NumPy, pandas, matplotlib, scikit-learn ≥ 1.5 |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุก cell + คำตอบคำถามท้ายข้อ (เขียนใน text cell) |

### วัตถุประสงค์

1. คำนวณ MAE, MSE, RMSE และ R² ด้วยมือ/NumPy และตรวจสอบกับ scikit-learn ได้
2. สร้าง Simple Linear Regression จากศูนย์ด้วย Normal Equation และ Gradient Descent แล้วเปรียบเทียบกับ `LinearRegression` ได้
3. ทำ Standardization อย่างถูกต้อง (fit กับ train เท่านั้น) และใช้ `Pipeline` ได้
4. สร้างและเปรียบเทียบ regressor หลายแบบ (Linear, Ridge, Lasso, SVR, Random Forest, Gradient Boosting) ด้วยตาราง metric ได้
5. วิเคราะห์ residual, ตรวจ overfitting จาก train/test error และตีความค่าสัมประสิทธิ์ของโมเดลเชิงเส้นได้

> [!NOTE]
> เนื้อหาประกอบอยู่ใน [บทที่ 4](/module/04) ให้รันทุก cell จากบนลงล่าง ส่วนที่มี `# TODO:` คือจุดที่ต้องเติมโค้ด (โค้ดเริ่มต้นรันผ่านได้ แต่ผลลัพธ์ยังไม่ถูกต้องจนกว่าจะเติม) ผลลัพธ์ที่คาดหวังได้มาจากการรันเฉลยด้วย scikit-learn 1.9 ตัวเลขทศนิยมตำแหน่งท้าย ๆ อาจต่างเล็กน้อยตามเวอร์ชัน

## ข้อที่ 0: เตรียมสภาพแวดล้อม

### โจทย์

import ไลบรารี โหลดชุดข้อมูล California Housing จาก `sklearn.datasets` (Colab ดาวน์โหลดให้อัตโนมัติ ไม่ต้องอัปโหลดไฟล์) และแบ่ง train/test 80:20 ด้วย `random_state=42` เพื่อให้ผลตรงกับเฉลย

### โค้ดเริ่มต้น

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target          # y = MedHouseVal (หน่วย $100,000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("train:", X_train.shape, " test:", X_test.shape)
X_train.head()
```

### ผลลัพธ์ที่คาดหวัง

```output
train: (16512, 8)  test: (4128, 8)
```

(ตามด้วยตาราง 5 แถวแรกของ `X_train` ที่มี 8 คอลัมน์ `MedInc` … `Longitude`)

## ข้อที่ 1: คำนวณ Metric ด้วยมือ

### โจทย์

กำหนดราคาจริงและราคาที่โมเดลทำนาย (หน่วย \$100,000) ของบ้าน 5 เขต

| เขต | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $y$ | 2.5 | 3.0 | 4.0 | 5.5 | 6.0 |
| $\hat{y}$ | 3.0 | 2.5 | 4.5 | 5.0 | 7.0 |

1. คำนวณ MAE, MSE, RMSE และ R² **ด้วยมือ** (แสดงตาราง $e$, $|e|$, $e^2$, $(y-\bar{y})^2$ ใน text cell)
2. เขียนฟังก์ชัน NumPy `my_mae`, `my_mse`, `my_rmse`, `my_r2` ตามสูตรในบทเรียน แล้วเทียบกับ scikit-learn

$$ \text{MAE}=\frac{1}{n}\sum|y_i-\hat{y}_i| \qquad \text{MSE}=\frac{1}{n}\sum(y_i-\hat{y}_i)^2 \qquad R^2 = 1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2} $$

### โค้ดเริ่มต้น

```python
y_true = np.array([2.5, 3.0, 4.0, 5.5, 6.0])
y_hat  = np.array([3.0, 2.5, 4.5, 5.0, 7.0])

def my_mae(y, y_pred):
    return None  # TODO: ค่าเฉลี่ยของ |y - y_pred|

def my_mse(y, y_pred):
    return None  # TODO: ค่าเฉลี่ยของ (y - y_pred)**2

def my_rmse(y, y_pred):
    return None  # TODO: รากที่สองของ MSE (เรียกใช้ my_mse)

def my_r2(y, y_pred):
    return None  # TODO: 1 - SS_res / SS_tot

print("MAE :", my_mae(y_true, y_hat),  "| sklearn:", mean_absolute_error(y_true, y_hat))
print("MSE :", my_mse(y_true, y_hat),  "| sklearn:", mean_squared_error(y_true, y_hat))
print("RMSE:", my_rmse(y_true, y_hat), "| sklearn:", root_mean_squared_error(y_true, y_hat))
print("R2  :", my_r2(y_true, y_hat),   "| sklearn:", r2_score(y_true, y_hat))
```

### ผลลัพธ์ที่คาดหวัง

```output
MAE : 0.6 | sklearn: 0.6
MSE : 0.4 | sklearn: 0.4
RMSE: 0.6324555320336759 | sklearn: 0.6324555320336759
R2  : 0.7849462365591398 | sklearn: 0.7849462365591398
```

### คำถามท้ายข้อ

1. RMSE (0.632) มีหน่วยอะไร และแปลเป็นเงินดอลลาร์ได้เท่าไร?
2. ถ้าเขตที่ 5 ทายเป็น 9.0 แทน 7.0 ค่าใดระหว่าง MAE กับ RMSE จะเพิ่มขึ้นในสัดส่วนที่มากกว่า เพราะอะไร? (ลองแก้ `y_hat` แล้วรันดู)
3. ถ้าโมเดลทาย $\hat{y} = \bar{y} = 4.2$ ทุกเขต R² จะเท่ากับเท่าไร?

## ข้อที่ 2: Simple Linear Regression ด้วย Normal Equation

### โจทย์

ใช้ feature เดียวคือ `MedInc` (รายได้มัธยฐาน) ทำนายราคาบ้าน

1. สร้างเมทริกซ์ `X_b` โดยเติมคอลัมน์ 1 ไว้หน้า `x_train`
2. คำนวณ $\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ ด้วย `np.linalg.inv` และ `@`
3. ฝึก `LinearRegression` กับ feature เดียวกัน เทียบค่า $\theta_0, \theta_1$ แล้วประเมินบน test set

### โค้ดเริ่มต้น

```python
x_train = X_train[["MedInc"]].to_numpy()     # shape (16512, 1)
x_test = X_test[["MedInc"]].to_numpy()

X_b = np.c_[np.ones((len(x_train), 1)), x_train]   # เติม x0 = 1
theta_ne = None  # TODO: (XᵀX)⁻¹ Xᵀy  → ใช้ np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train.to_numpy()
print("Normal Equation  θ =", theta_ne.round(4) if theta_ne is not None else None)

lin1 = LinearRegression()
lin1.fit(x_train, y_train)
print("LinearRegression θ =", np.r_[lin1.intercept_, lin1.coef_].round(4))

y_pred1 = lin1.predict(x_test)
print(f"test RMSE = {root_mean_squared_error(y_test, y_pred1):.4f}  R2 = {r2_score(y_test, y_pred1):.4f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
Normal Equation  θ = [0.4446 0.4193]
LinearRegression θ = [0.4446 0.4193]
test RMSE = 0.8421  R2 = 0.4589
```

### คำถามท้ายข้อ

1. เขียนสมการที่ได้ในรูป $\hat{y} = \theta_0 + \theta_1 \cdot \text{MedInc}$ และตีความ $\theta_1 = 0.4193$ เป็นภาษาธุรกิจ (MedInc หน่วย \$10,000, ราคาหน่วย \$100,000)
2. ใช้ `MedInc` ตัวเดียวได้ R² ≈ 0.46 แปลว่าอะไร? สอดคล้องกับค่า correlation 0.69 ระหว่าง `MedInc` กับราคาอย่างไร (ลองคำนวณ $0.69^2$)

## ข้อที่ 3: Gradient Descent จากศูนย์ และผลของ Learning Rate

### โจทย์

1. Standardize `x_train` ด้วย $\mu, \sigma$ ของ train แล้วเติมคอลัมน์ 1
2. เติมโค้ดใน `batch_gd` ให้คำนวณ gradient $\nabla\text{MSE} = \frac{2}{m}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta}-\mathbf{y})$ และปรับ $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\text{MSE}$
3. รันด้วย $\eta$ = 0.001, 0.01, 0.1, 0.5 พร้อมพล็อต train MSE ต่อ epoch และทดลอง $\eta$ = 1.05
4. แปลง $\theta$ ที่ได้กลับเป็นหน่วยเดิม: slope $= \theta_1/\sigma$, intercept $= \theta_0 - \theta_1\mu/\sigma$ แล้วเทียบกับข้อที่ 2
5. ลองใช้ `SGDRegressor` ใน `Pipeline` กับ feature เดียวกัน

### โค้ดเริ่มต้น

```python
mu, sigma = x_train.mean(), x_train.std()
x_s = (x_train - mu) / sigma                   # standardize ด้วยค่าของ train
X_bs = np.c_[np.ones((len(x_s), 1)), x_s]
y_np = y_train.to_numpy()
m = len(X_bs)

def batch_gd(eta, n_epochs=100):
    theta = np.zeros(2)
    history = []
    for epoch in range(n_epochs):
        gradients = np.zeros(2)  # TODO: 2 / m * X_bs.T @ (X_bs @ theta - y_np)
        theta = theta            # TODO: theta - eta * gradients
        history.append(np.mean((X_bs @ theta - y_np) ** 2))
    return theta, history

plt.figure(figsize=(7, 4))
for eta in [0.001, 0.01, 0.1, 0.5]:
    theta, history = batch_gd(eta)
    plt.plot(history, label=f"eta={eta}")
    print(f"eta={eta:<6} θ={np.round(theta, 4)}  final MSE={history[-1]:.4f}")
plt.xlabel("epoch"); plt.ylabel("train MSE"); plt.legend(); plt.title("Batch GD: effect of learning rate")
plt.show()

theta_gd, _ = batch_gd(0.1)
slope = theta_gd[1] / sigma
intercept = 0.0  # TODO: theta_gd[0] - theta_gd[1] * mu / sigma
print("GD → หน่วยเดิม: intercept =", round(intercept, 4), " slope =", round(slope, 4))

_, hist_big = batch_gd(1.05, n_epochs=100)
print("eta=1.05 final MSE =", f"{hist_big[-1]:.4g}")

sgd = make_pipeline(StandardScaler(), SGDRegressor(eta0=0.01, penalty=None, random_state=42))
sgd.fit(x_train, y_train)
print("SGDRegressor test RMSE =", round(root_mean_squared_error(y_test, sgd.predict(x_test)), 4))
```

### ผลลัพธ์ที่คาดหวัง

```output
eta=0.001  θ=[0.3759 0.1449]  final MSE=4.0029
eta=0.01   θ=[1.7972 0.6926]  final MSE=0.7859
eta=0.1    θ=[2.0719 0.7985]  final MSE=0.6991
eta=0.5    θ=[2.0719 0.7985]  final MSE=0.6991
GD → หน่วยเดิม: intercept = 0.4446  slope = 0.4193
eta=1.05 final MSE = 9.363e+08
SGDRegressor test RMSE = 0.8422
```

(และกราฟ 4 เส้น: เส้น eta=0.001 ลดลงช้ามาก เส้น 0.01 ลดลงแต่ยังไม่ถึงค่าต่ำสุด ส่วนเส้น 0.1 และ 0.5 ลดลงจนคงที่ที่ MSE ≈ 0.699 โดย 0.5 เร็วที่สุด)

### คำถามท้ายข้อ

1. ทำไม $\eta$ = 0.001 จึงยังมี MSE สูงหลัง 100 epochs และ $\eta$ = 1.05 จึง MSE พุ่งสูงขึ้นแทนที่จะลดลง?
2. ค่า slope/intercept ที่แปลงกลับจาก GD ตรงกับ Normal Equation ในข้อ 2 หรือไม่? ถ้าไม่ standardize `x` ก่อน จะเกิดอะไรกับการเลือก $\eta$?
3. เปรียบเทียบ Batch GD, SGD และ Mini-batch GD ว่าแบบใดเหมาะกับข้อมูลหลายล้านแถว เพราะอะไร

## ข้อที่ 4: สำรวจข้อมูลและ Standardization ที่ถูกต้อง

### โจทย์

1. นับจำนวนเขตใน train ที่ราคาชนเพดาน (≥ 5.0)
2. สร้าง `StandardScaler` → `fit_transform` กับ `X_train` และ `transform` กับ `X_test` (ห้าม fit กับ test)
3. ตรวจ mean/std หลัง scale ของ train และ test
4. คำนวณ z-score ของ `MedInc` แถวแรกใน test ด้วยมือ (ใช้ mean/std ของ **train**, `ddof=0`) แล้วเทียบกับผลของ scaler

### โค้ดเริ่มต้น

```python
n_capped = 0  # TODO: (y_train >= 5.0).sum()
print("เขตที่ราคา ≥ 5.0 ใน train:", n_capped)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = None  # TODO: ใช้ scaler.transform กับ X_test (ห้าม fit ใหม่)
print("train mean:", X_train_s.mean(axis=0).round(3))
print("train std :", X_train_s.std(axis=0).round(3))
if X_test_s is not None:
    print("test  mean:", X_test_s.mean(axis=0).round(3))

manual = 0.0  # TODO: (X_test["MedInc"].iloc[0] - X_train["MedInc"].mean()) / X_train["MedInc"].std(ddof=0)
print("z ของ MedInc แถวแรกใน test (คำนวณเอง):", round(manual, 4),
      " scaler:", round(X_test_s[0, 0], 4) if X_test_s is not None else None)
```

### ผลลัพธ์ที่คาดหวัง

```output
เขตที่ราคา ≥ 5.0 ใน train: 808
train mean: [-0. -0.  0. -0.  0.  0.  0.  0.]
train std : [1. 1. 1. 1. 1. 1. 1. 1.]
test  mean: [-0.026  0.012 -0.013 -0.    -0.004 -0.011 -0.026  0.031]
z ของ MedInc แถวแรกใน test (คำนวณเอง): -1.1551  scaler: -1.1551
```

### คำถามท้ายข้อ

1. ทำไม mean ของ test หลัง scale **ไม่เท่ากับ 0 พอดี** และนี่เป็นเรื่องผิดปกติหรือไม่?
2. ถ้า fit scaler กับข้อมูลทั้งหมดก่อนแบ่ง train/test จะเกิดปัญหาอะไร? `Pipeline` ช่วยป้องกันได้อย่างไร?
3. ข้อมูลที่ราคาถูก cap ที่ 5.0 จะส่งผลต่อโมเดลอย่างไร (คาดการณ์ไว้ก่อน แล้วไปตรวจในข้อที่ 6)

## ข้อที่ 5: เปรียบเทียบ Regressor หลายแบบ

### โจทย์

เติมโมเดลใน dictionary `models` ให้ครบ 6 แบบ — โมเดลที่ไวต่อสเกลต้องอยู่ใน `make_pipeline(StandardScaler(), ...)`

| ชื่อ | โมเดล |
|---|---|
| `Ridge(alpha=1)` | `Ridge(alpha=1.0)` + scaler |
| `Lasso(alpha=0.01)` | `Lasso(alpha=0.01)` + scaler |
| `SVR-RBF (5k)` | `SVR(kernel="rbf", C=10, epsilon=0.1)` + scaler (โค้ดสุ่ม train มา 5,000 แถวให้แล้ว เพราะ SVR แบบ kernel ช้ามากกับข้อมูล 16,512 แถว) |
| `RandomForest` | `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` (ไม่ต้อง scale) |
| `HistGBoosting` | `HistGradientBoostingRegressor(random_state=42)` (ไม่ต้อง scale) |

### โค้ดเริ่มต้น

```python
models = {
    "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge(alpha=1)":    None,  # TODO
    "Lasso(alpha=0.01)": None,  # TODO
    "SVR-RBF (5k)":      None,  # TODO
    "RandomForest":      None,  # TODO
    "HistGBoosting":     None,  # TODO
}

rows, fitted = [], {}
for name, model in models.items():
    if model is None:                       # ข้ามโมเดลที่ยังไม่ได้เติม
        continue
    Xt, yt = X_train, y_train
    if name.startswith("SVR"):              # SVR ฝึกด้วย 5,000 แถว
        Xt = X_train.sample(5000, random_state=42)
        yt = y_train.loc[Xt.index]
    model.fit(Xt, yt)
    fitted[name] = model
    y_pred = model.predict(X_test)
    rows.append({"model": name,
                 "train_RMSE": root_mean_squared_error(yt, model.predict(Xt)),
                 "test_RMSE": root_mean_squared_error(y_test, y_pred),
                 "test_MAE": mean_absolute_error(y_test, y_pred),
                 "test_R2": r2_score(y_test, y_pred)})

results = pd.DataFrame(rows).set_index("model").round(3)
print(results.sort_values("test_RMSE"))
```

### ผลลัพธ์ที่คาดหวัง

```output
                   train_RMSE  test_RMSE  test_MAE  test_R2
model                                                      
HistGBoosting           0.401      0.465     0.311    0.835
RandomForest            0.188      0.506     0.328    0.805
SVR-RBF (5k)            0.553      0.601     0.403    0.724
Lasso(alpha=0.01)       0.723      0.740     0.535    0.582
LinearRegression        0.720      0.746     0.533    0.576
Ridge(alpha=1)          0.720      0.746     0.533    0.576
```

(Random Forest และ HistGradientBoosting อาจใช้เวลาฝึกหลายวินาทีถึงหลายสิบวินาทีใน Colab)

### คำถามท้ายข้อ

1. แปลง test RMSE ของโมเดลที่ดีที่สุดและของ `LinearRegression` เป็นดอลลาร์ ต่างกันเท่าไร?
2. ทำไมโมเดลเชิงเส้นทั้งสามแบบ (Linear, Ridge, Lasso) ได้ผลใกล้เคียงกันมาก แต่ Tree ensemble ดีกว่าชัดเจน? (ใบ้: ดูแผนที่ราคาบ้านในบทเรียน)
3. โมเดลใดมีช่องว่างระหว่าง train_RMSE กับ test_RMSE มากที่สุด แปลว่าอะไร?
4. ถ้าต้องใช้ SVR กับข้อมูลทั้งหมด 16,512 แถว (หรือมากกว่านั้น) จะเลือกทางใดได้บ้าง?

## ข้อที่ 6: Residual Analysis

### โจทย์

1. พล็อต residual plot ($\hat{y}$ vs $y-\hat{y}$) ของ `LinearRegression` เทียบกับโมเดลที่ test RMSE ต่ำที่สุด
2. คำนวณ mean residual ของ (ก) test ทั้งหมด (ข) เขตที่ราคาถูก cap ($y \ge 5.0$) (ค) เขตอื่น

### โค้ดเริ่มต้น

```python
best_name = results["test_RMSE"].idxmin()
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, name in zip(axes, ["LinearRegression", best_name]):
    y_pred = fitted[name].predict(X_test)
    residual = y_test - y_pred               # e = y − ŷ
    ax.scatter(y_pred, residual, s=4, alpha=0.3)
    ax.axhline(0, color="red")
    ax.set(title=name, xlabel="predicted", ylabel="residual")

    capped = y_test >= 5.0
    mean_capped = 0.0  # TODO: residual[capped].mean()
    mean_other = 0.0   # TODO: residual[~capped].mean()
    print(f"{name:<16} mean residual: ทั้งหมด={residual.mean():.3f} | เขตที่ถูก cap={mean_capped:.3f} | เขตอื่น={mean_other:.3f}")
plt.tight_layout()
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
LinearRegression mean residual: ทั้งหมด=0.003 | เขตที่ถูก cap=1.153 | เขตอื่น=-0.050
HistGBoosting    mean residual: ทั้งหมด=-0.008 | เขตที่ถูก cap=0.610 | เขตอื่น=-0.037
```

(กราฟ: residual ของ Linear เป็นรูปพัดและมีเส้นทแยงลงจากเขตที่ราคาถูก cap บางจุดทำนายติดลบหรือเกิน 10 ส่วนของ HistGB กระจุกแน่นรอบ 0 กว่า)

### คำถามท้ายข้อ

1. Mean residual ของทั้ง test set ใกล้ 0 แต่ของเขตที่ถูก cap เป็นบวกมาก แปลว่าโมเดลทำนายบ้านราคาสูง "ต่ำไป" หรือ "สูงไป"? ทำไมจึงเกิดขึ้น
2. เส้นทแยงลงในกราฟเกิดจากอะไร (ใบ้: ถ้า $y = 5.0$ คงที่ residual $= 5.0 - \hat{y}$ เป็นฟังก์ชันอะไรของ $\hat{y}$)
3. เสนอวิธีแก้ปัญหาข้อมูลที่ถูก cap อย่างน้อย 2 วิธี

## ข้อที่ 7: ตีความค่าสัมประสิทธิ์และผลของ Lasso

### โจทย์

1. ฝึก `LinearRegression` แบบ **ไม่ scale** เพื่อดูค่าสัมประสิทธิ์ในหน่วยเดิม และดึงค่าสัมประสิทธิ์ของ pipeline ที่ scale แล้ว (`fitted["LinearRegression"][-1].coef_`) มาวางคู่กัน เรียงตามขนาดค่าสัมบูรณ์ของค่าที่ scale แล้ว
2. วนค่า `alpha` ของ Lasso = 0.001, 0.01, 0.1, 0.5 นับจำนวนสัมประสิทธิ์ที่เป็น 0 และ test RMSE

### โค้ดเริ่มต้น

```python
lin_raw = None  # TODO: LinearRegression().fit(X_train, y_train)
if lin_raw is not None:
    coef_raw = pd.Series(lin_raw.coef_, index=X.columns)
    coef_std = pd.Series(fitted["LinearRegression"][-1].coef_, index=X.columns)
    table = pd.DataFrame({"coef_raw": coef_raw, "coef_scaled": coef_std})
    print(table.reindex(coef_std.abs().sort_values(ascending=False).index).round(4))

for alpha in [0.001, 0.01, 0.1, 0.5]:
    lasso = make_pipeline(StandardScaler(), Lasso(alpha=alpha)).fit(X_train, y_train)
    n_zero = 0  # TODO: นับจำนวน lasso[-1].coef_ ที่เท่ากับ 0 → int(np.sum(lasso[-1].coef_ == 0))
    print(f"Lasso alpha={alpha:<5}  coef=0: {n_zero} ตัว  test RMSE={root_mean_squared_error(y_test, lasso.predict(X_test)):.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
            coef_raw  coef_scaled
Latitude     -0.4198      -0.8969
Longitude    -0.4337      -0.8698
MedInc        0.4487       0.8544
AveBedrms     0.7831       0.3393
AveRooms     -0.1233      -0.2944
HouseAge      0.0097       0.1225
AveOccup     -0.0035      -0.0408
Population   -0.0000      -0.0023
Lasso alpha=0.001  coef=0: 0 ตัว  test RMSE=0.745
Lasso alpha=0.01   coef=0: 1 ตัว  test RMSE=0.740
Lasso alpha=0.1    coef=0: 5 ตัว  test RMSE=0.824
Lasso alpha=0.5    coef=0: 7 ตัว  test RMSE=0.970
```

### คำถามท้ายข้อ

1. ตีความ `coef_raw` ของ `MedInc` (0.4487) เป็นจำนวนดอลลาร์ และระบุเงื่อนไขสำคัญของการตีความนี้
2. ทำไม `Population` มี `coef_raw` ≈ 0 จึง **สรุปไม่ได้** ว่า feature นี้ไม่สำคัญ? ควรใช้คอลัมน์ใดเปรียบเทียบความสำคัญแทน
3. `AveRooms` ติดลบแต่ `AveBedrms` เป็นบวก ขัดกับสามัญสำนึกหรือไม่? อธิบายด้วยแนวคิด multicollinearity
4. เมื่อ `alpha` เพิ่มขึ้น จำนวนสัมประสิทธิ์ที่เป็นศูนย์และ test RMSE เปลี่ยนไปอย่างไร ค่า alpha ใดเหมาะที่สุด และควรเลือก alpha ด้วยวิธีใดแทนการดู test set?
5. **สรุปภาพรวม:** หากต้องเสนอโมเดลให้บริษัทอสังหาริมทรัพย์ใช้ประเมินราคา จะเลือกโมเดลใด ระหว่าง "แม่นที่สุด" กับ "อธิบายได้ง่ายที่สุด" ให้เหตุผลโดยอ้างตัวเลขจากข้อ 5–7

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | รายละเอียดในใบงานนี้ | คะแนน |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | อธิบายความหมายของ MAE/MSE/RMSE/R², learning rate, regularization, data leakage ในคำถามท้ายข้อ | 2 |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | เลือกใช้ scaler กับโมเดลที่ถูกต้อง, จัดการ SVR กับข้อมูลใหญ่, เลือกโมเดลสุดท้ายพร้อมเหตุผล (ข้อ 5, 7) | 2 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | แบ่งข้อมูล → scale ใน Pipeline → ฝึก → ประเมิน → residual analysis ครบและรันได้ตามลำดับ | 2 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | คำนวณ metric ด้วยมือ (ข้อ 1), Normal Equation/GD (ข้อ 2–3), ตาราง metric (ข้อ 5) ได้ผลตรงกับที่คาดหวัง | 3 |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | ตีความค่าสัมประสิทธิ์ residual และข้อจำกัดจากข้อมูลที่ถูก cap (ข้อ 6–7) | 1 |
| **รวม** | | **10** |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab04_<รหัสนักศึกษา>.ipynb` ที่รันครบทุก cell (Runtime → Run all) โดยไม่มี error และทุก `# TODO:` ถูกเติมแล้ว
2. คำตอบคำถามท้ายข้อทุกข้อ เขียนใน text cell ใต้ข้อนั้น ๆ (ข้อที่ 1 ให้แสดงตารางการคำนวณด้วยมือ)
3. กราฟอย่างน้อย 2 รูป: เส้น MSE ของ Gradient Descent (ข้อ 3) และ residual plot (ข้อ 6)
4. ส่งผ่านระบบ LMS ของรายวิชาภายในกำหนดเวลาที่ผู้สอนประกาศ
