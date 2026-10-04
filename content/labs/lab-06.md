## ข้อมูลทั่วไปของใบงาน

| รายการ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 06: ออกแบบและฝึก MLP ด้วย Keras |
| งานตาม มคอ.3 | แบบฝึกหัด 6: neural network architecture |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 9 |
| CLO/LLO | CLO3 / LLO3-1 อธิบายโครงสร้างและหลักการทำงานของ Neural Networks |
| เครื่องมือ | Google Colab (เปิด GPU: *Runtime → Change runtime type → T4 GPU*), NumPy, matplotlib, TensorFlow/Keras 3, scikit-learn |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ + ตอบคำถามท้ายข้อในเซลล์ markdown |

### วัตถุประสงค์

1. คำนวณ forward pass ของเครือข่าย 2-3-1 ได้ทั้งด้วยมือและด้วย NumPy
2. นับจำนวนพารามิเตอร์ของ MLP หลายสถาปัตยกรรมได้ถูกต้อง
3. เปรียบเทียบ activation function และอนุพันธ์ของมันผ่านกราฟ
4. ออกแบบ สร้าง ฝึก และประเมิน MLP ด้วย Keras บน Fashion-MNIST พร้อมอ่าน learning curves
5. ทดลองปรับจำนวน hidden units, Dropout และ optimizer แล้วสรุปผลเป็นตาราง
6. วาดและอธิบายสถาปัตยกรรมเครือข่ายที่เหมาะกับโจทย์ต่าง ๆ

## ข้อที่ 0: เตรียมสภาพแวดล้อม

เซลล์นี้ import ไลบรารีและโหลด Fashion-MNIST (ภาพเสื้อผ้า 28×28 จำนวน 70,000 ภาพ 10 คลาส) ถ้ารันนอก Colab แล้วไม่มี TensorFlow จะโหลดข้อมูลจาก OpenML แทน และข้ามเซลล์ Keras ไปใช้ข้อที่ 6 (scikit-learn) ได้

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

try:
    import tensorflow as tf
    import keras
    HAS_KERAS = True
    print("TensorFlow", tf.__version__, "| Keras", keras.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
except ImportError:
    HAS_KERAS = False
    print("ไม่พบ TensorFlow/Keras — ใช้ข้อที่ 6 (scikit-learn) แทนข้อ 4-5")

if HAS_KERAS:
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
else:
    from sklearn.datasets import fetch_openml
    X_all, y_all = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
    X_all = X_all.reshape(-1, 28, 28)
    y_all = y_all.astype(int)
    X_train, X_test, y_train, y_test = X_all[:60000], X_all[60000:], y_all[:60000], y_all[60000:]

X_train = X_train / 255.0      # scale pixel เป็น [0, 1]
X_test = X_test / 255.0
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("10 label แรก:", y_train[:10])
```

บรรทัดแรก ๆ จะแสดงเวอร์ชันของ TensorFlow/Keras และรายการ GPU (ถ้าไม่ได้เปิด GPU จะเป็น `[]`) ตามด้วย

```output
X_train: (60000, 28, 28) | X_test: (10000, 28, 28)
10 label แรก: [9 0 0 3 0 2 7 2 5 5]
```

```python
# ดูตัวอย่างภาพ
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, img, label in zip(axes.ravel(), X_train[:10], y_train[:10]):
    ax.imshow(img, cmap="binary")
    ax.set_title(class_names[label], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

## ข้อที่ 1: Forward pass ของเครือข่าย 2-3-1

### โจทย์

เครือข่ายมี input 2 ตัว hidden layer 3 neuron (ReLU) และ output 1 neuron (sigmoid) กำหนดค่าดังนี้

- $\mathbf{x} = [1.0,\ 2.0]$, คำตอบจริง $y = 1$
- $\mathbf{W}_1 = \begin{bmatrix} 0.2 & -0.5 & 1.0 \\ 0.4 & 0.3 & -1.0 \end{bmatrix}$, $\mathbf{b}_1 = [0.1,\ 0.0,\ 0.5]$
- $\mathbf{W}_2 = \begin{bmatrix} 0.5 \\ -1.0 \\ 2.0 \end{bmatrix}$, $b_2 = -0.2$

1. **คำนวณด้วยมือ** (เขียนในเซลล์ markdown หรือกระดาษแล้วถ่ายรูปแนบ): $\mathbf{z}_1 = \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1$, $\mathbf{h} = \text{ReLU}(\mathbf{z}_1)$, $z_2 = \mathbf{h}\mathbf{W}_2 + b_2$, $\hat{y} = \sigma(z_2)$ และ binary cross-entropy loss
2. เติมโค้ด NumPy ให้ได้ผลตรงกับที่คำนวณด้วยมือ
3. นับจำนวนพารามิเตอร์ทั้งหมดของเครือข่าย

### โค้ดเริ่มต้น

```python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x  = np.array([1.0, 2.0])
W1 = np.array([[0.2, -0.5,  1.0],
               [0.4,  0.3, -1.0]])
b1 = np.array([0.1, 0.0, 0.5])
W2 = np.array([[0.5], [-1.0], [2.0]])
b2 = np.array([-0.2])
y_true = 1

z1 = np.zeros(3)       # TODO: z1 = x @ W1 + b1
h = np.zeros(3)        # TODO: h = relu(z1)
z2 = np.zeros(1)       # TODO: z2 = h @ W2 + b2
y_hat = np.zeros(1)    # TODO: y_hat = sigmoid(z2)
loss = 0.0             # TODO: binary cross-entropy ของ y_true กับ y_hat[0]
n_params = 0           # TODO: รวมขนาดของ W1, b1, W2, b2 (ใช้ .size)

print("z1    =", z1)
print("h     =", h)
print("z2    =", z2)
print("y_hat =", y_hat)
print(f"BCE loss = {loss:.4f}")
print("จำนวนพารามิเตอร์ =", n_params)
```

### ผลลัพธ์ที่คาดหวัง

```output
z1    = [ 1.1  0.1 -0.5]
h     = [1.1 0.1 0. ]
z2    = [0.25]
y_hat = [0.5622]
BCE loss = 0.5759
จำนวนพารามิเตอร์ = 13
```

### คำถามท้ายข้อ

1. neuron ตัวที่ 3 ของ hidden layer มีส่วนต่อ output หรือไม่ เพราะอะไร (สัมพันธ์กับปัญหา dying ReLU อย่างไร)
2. ถ้าต้องการให้ $\hat{y}$ เข้าใกล้ 1 มากขึ้น gradient descent ควรปรับ $b_2$ ให้เพิ่มหรือลด (ใช้ $\partial L / \partial z_2 = \hat{y} - y$)

## ข้อที่ 2: นับจำนวนพารามิเตอร์ของสถาปัตยกรรมต่าง ๆ

### โจทย์

1. เขียนฟังก์ชัน `dense_params(n_in, n_out)` ตามสูตร $n_{in} \times n_{out} + n_{out}$
2. เขียน `mlp_params(layer_sizes)` ที่รวมพารามิเตอร์ของทุก Dense layer เมื่อ `layer_sizes` คือจำนวน neuron ของแต่ละชั้นรวม input
3. คำนวณสถาปัตยกรรม D ที่ใส่ `BatchNormalization` หลัง hidden layer ทั้งสองชั้น (BatchNorm มี 4 ค่าต่อ neuron)

### โค้ดเริ่มต้น

```python
def dense_params(n_in, n_out):
    return 0  # TODO: weights + biases

def mlp_params(layer_sizes):
    total = 0
    # TODO: วนคู่ (n_in, n_out) ด้วย zip(layer_sizes[:-1], layer_sizes[1:]) แล้วบวก dense_params
    return total

architectures = {
    "A: Iris 4-8-3": [4, 8, 3],
    "B: California housing 8-30-1": [8, 30, 1],
    "C: Fashion-MNIST 784-128-10": [784, 128, 10],
    "D: Fashion-MNIST 784-300-100-10": [784, 300, 100, 10],
    "E: Breast cancer 30-16-8-1": [30, 16, 8, 1],
}
for name, sizes in architectures.items():
    print(f"{name:<34} params = {mlp_params(sizes):>9,}")

bn_params = 0  # TODO: จำนวนพารามิเตอร์ของ BatchNormalization หลัง Dense(300) และ Dense(100)
print("D + BatchNorm หลัง hidden ทั้ง 2 ชั้น =", f"{mlp_params([784, 300, 100, 10]) + bn_params:,}")
```

### ผลลัพธ์ที่คาดหวัง

```output
A: Iris 4-8-3                      params =        67
B: California housing 8-30-1       params =       301
C: Fashion-MNIST 784-128-10        params =   101,770
D: Fashion-MNIST 784-300-100-10    params =   266,610
E: Breast cancer 30-16-8-1         params =       641
D + BatchNorm หลัง hidden ทั้ง 2 ชั้น = 268,210
```

### คำถามท้ายข้อ

1. ในสถาปัตยกรรม D ชั้นใดมีพารามิเตอร์มากที่สุด คิดเป็นกี่ % ของทั้งหมด และเพราะอะไร
2. ในพารามิเตอร์ของ BatchNorm 1,600 ตัว มีกี่ตัวที่ฝึกได้ (trainable) และกี่ตัวที่ฝึกไม่ได้ (non-trainable) แต่ละกลุ่มคืออะไร
3. ถ้าเพิ่ม `Dropout(0.3)` หลังทุก hidden layer จำนวนพารามิเตอร์เปลี่ยนหรือไม่

## ข้อที่ 3: Activation functions และอนุพันธ์

### โจทย์

1. เขียนฟังก์ชัน sigmoid, tanh, ReLU และ Leaky ReLU ($\alpha = 0.1$) ด้วย NumPy
2. วาดกราฟของฟังก์ชัน (ซ้าย) และอนุพันธ์ (ขวา) ในช่วง $z \in [-5, 5]$ — โค้ดใช้ `np.gradient` หาอนุพันธ์เชิงตัวเลขให้แล้ว
3. หาค่าอนุพันธ์สูงสุดของแต่ละฟังก์ชัน

### โค้ดเริ่มต้น

```python
def sigmoid_f(z):
    return np.zeros_like(z)   # TODO: 1 / (1 + e^-z)

def tanh_f(z):
    return np.zeros_like(z)   # TODO: ใช้ np.tanh

def relu_f(z):
    return np.zeros_like(z)   # TODO: max(0, z)

def leaky_relu_f(z, alpha=0.1):
    return np.zeros_like(z)   # TODO: z ถ้า z > 0 ไม่เช่นนั้น alpha * z (ใช้ np.where)

z = np.linspace(-5, 5, 1001)
funcs = {"sigmoid": sigmoid_f, "tanh": tanh_f, "ReLU": relu_f, "Leaky ReLU": leaky_relu_f}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name, f in funcs.items():
    a = f(z)
    axes[0].plot(z, a, label=name)
    axes[1].plot(z, np.gradient(a, z), label=name)
axes[0].set_title("Activation functions")
axes[1].set_title("Derivatives")
for ax in axes:
    ax.grid(True); ax.legend(); ax.set_xlabel("z")
plt.tight_layout()
plt.show()

z_test = np.array([-2.0, 0.0, 2.0])
for name, f in funcs.items():
    d_max = np.gradient(f(z), z).max()
    print(f"{name:<10} f(-2,0,2) = {f(z_test)}   อนุพันธ์สูงสุด ≈ {d_max:.3f}")
```

### ผลลัพธ์ที่คาดหวัง

กราฟซ้าย: sigmoid เป็นรูปตัว S ระหว่าง 0–1, tanh เป็นรูปตัว S ระหว่าง −1–1, ReLU เป็นศูนย์ฝั่งลบและเส้นตรงฝั่งบวก, Leaky ReLU มีความชันเล็กน้อยฝั่งลบ กราฟขวา: อนุพันธ์ของ sigmoid เป็นระฆังสูงสุด 0.25 ที่ $z=0$, tanh สูงสุด 1 ที่ $z=0$ และลดลงเร็ว ส่วน ReLU เป็นขั้นบันได 0 → 1

```output
sigmoid    f(-2,0,2) = [0.1192 0.5    0.8808]   อนุพันธ์สูงสุด ≈ 0.250
tanh       f(-2,0,2) = [-0.964  0.     0.964]   อนุพันธ์สูงสุด ≈ 1.000
ReLU       f(-2,0,2) = [0. 0. 2.]   อนุพันธ์สูงสุด ≈ 1.000
Leaky ReLU f(-2,0,2) = [-0.2  0.   2. ]   อนุพันธ์สูงสุด ≈ 1.000
```

### คำถามท้ายข้อ

1. จากกราฟอนุพันธ์ อธิบายว่าทำไมเครือข่าย 10 ชั้นที่ใช้ sigmoid ทุกชั้นจึงฝึกยาก (คำนวณ $0.25^{10}$ ประกอบ)
2. ทำไม ReLU จึงเป็นค่าเริ่มต้นของ hidden layer และ Leaky ReLU แก้ข้อเสียใดของ ReLU
3. ฟังก์ชันใดเหมาะกับ output layer ของ (ก) binary classification (ข) multiclass classification (ค) regression

## ข้อที่ 4: ออกแบบและฝึก MLP ด้วย Keras บน Fashion-MNIST

### โจทย์

1. เติมฟังก์ชัน `build_mlp()` ให้สร้างโมเดล `keras.Input(shape=(28, 28))` → `Flatten` → hidden layers (`Dense` + ReLU และ `Dropout` ถ้า `dropout_rate > 0`) → `Dense(10, softmax)` แล้ว compile ด้วย loss ที่เหมาะกับ label จำนวนเต็ม 0–9
2. สร้างโมเดล 784-300-100-10 ดู `model.summary()` และตรวจว่าจำนวนพารามิเตอร์ตรงกับข้อ 2
3. ฝึกด้วย `validation_split=0.1` และ `EarlyStopping` วาด learning curves แล้วประเมินบน test set

### โค้ดเริ่มต้น

```python
def build_mlp(hidden_units=(300, 100), dropout_rate=0.0, optimizer="adam"):
    """สร้างและ compile MLP สำหรับ Fashion-MNIST"""
    if not HAS_KERAS:
        return None
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    for units in hidden_units:
        pass  # TODO: model.add(Dense(units, activation="relu")) และถ้า dropout_rate > 0 ให้ add Dropout(dropout_rate)
    # TODO: model.add(output layer 10 neuron, activation="softmax")
    # TODO: model.compile(loss=..., optimizer=optimizer, metrics=["accuracy"])
    return None  # TODO: เปลี่ยนเป็น return model

if HAS_KERAS:
    keras.utils.set_random_seed(42)
model = build_mlp(hidden_units=(300, 100))
if model is None:
    print("ยังไม่ได้สร้างโมเดล — ทำ TODO ใน build_mlp() ก่อน")
else:
    model.summary()
```

```python
EPOCHS = 30
history = None
if model is not None:
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                               restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32,
                        validation_split=0.1, callbacks=[early_stop])
```

```python
if history is not None:
    pd.DataFrame(history.history).plot(figsize=(8, 5), grid=True, xlabel="Epoch",
                                       style=["r--", "r-", "b--", "b-"])
    plt.title("Learning curves")
    plt.show()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"test loss = {test_loss:.4f} | test accuracy = {test_acc:.4f}")

    y_pred = model.predict(X_test[:5], verbose=0).argmax(axis=1)
    print("ทำนาย :", [class_names[i] for i in y_pred])
    print("คำตอบ :", [class_names[i] for i in y_test[:5]])
```

### ผลลัพธ์ที่คาดหวัง

`model.summary()` ต้องได้จำนวนพารามิเตอร์ตรงตามนี้ทุกครั้ง

```output
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flatten (Flatten)               │ (None, 784)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 300)            │       235,500 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 100)            │        30,100 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │         1,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 266,610 (1.02 MB)
 Trainable params: 266,610 (1.02 MB)
 Non-trainable params: 0 (0.00 B)
```

ตัวอย่างผลลัพธ์การฝึก (ค่าที่ได้จะต่างกันเล็กน้อยในแต่ละครั้ง): แต่ละ epoch มี 1,688 step (54,000 ÷ 32), val_accuracy อยู่ราว 0.84–0.86 ใน epoch แรกและขึ้นไปราว 0.88–0.90, EarlyStopping อาจหยุดก่อนครบ 30 epoch

```output
test loss ≈ 0.32-0.36 | test accuracy ≈ 0.87-0.89
ทำนาย : (ส่วนใหญ่ตรงกับคำตอบ)
คำตอบ : ['Ankle boot', 'Pullover', 'Trouser', 'Trouser', 'Shirt']
```

### คำถามท้ายข้อ

1. อธิบายความหมายของ `None` ใน Output Shape และเหตุผลที่ `Flatten` มีพารามิเตอร์ 0
2. ทำไมจึงใช้ `sparse_categorical_crossentropy` ไม่ใช่ `categorical_crossentropy` หรือ `binary_crossentropy` ถ้าต้องการใช้ `categorical_crossentropy` ต้องแปลง `y_train` อย่างไร
3. จาก learning curves ของคุณ เริ่มเห็นสัญญาณ overfitting ที่ epoch ใด (ดูจาก `loss` vs `val_loss`) และ EarlyStopping คืน weight ของ epoch ใด

## ข้อที่ 5: ทดลองปรับสถาปัตยกรรมและ optimizer

### โจทย์

ฝึกโมเดลตามการตั้งค่าในตาราง ครั้งละ 5 epoch (เพื่อประหยัดเวลา) บันทึก val_accuracy สูงสุด แล้วเติมคอลัมน์ "ข้อสังเกต" ในเซลล์ markdown ใต้ตาราง พร้อมเพิ่มการทดลองของตัวเองอย่างน้อย 1 แถว (เช่น `Dropout(0.5)`, 3 hidden layers หรือ `"nadam"`)

### โค้ดเริ่มต้น

```python
EXP_EPOCHS = 5
experiments = [
    # (ชื่อการทดลอง, hidden_units, dropout_rate, optimizer)
    ("baseline 300-100, adam", (300, 100), 0.0, "adam"),
    ("เล็ก 64",               (64,),      0.0, "adam"),
    ("ใหญ่ 512-256",          (512, 256), 0.0, "adam"),
    ("Dropout 0.3",           (300, 100), 0.3, "adam"),
    ("SGD",                   (300, 100), 0.0, "sgd"),
    ("RMSprop",               (300, 100), 0.0, "rmsprop"),
    # TODO: เพิ่มการทดลองของตัวเองอย่างน้อย 1 แถว
]

results = []
for name, hidden, drop, opt in experiments:
    if HAS_KERAS:
        keras.utils.set_random_seed(42)
    m = build_mlp(hidden_units=hidden, dropout_rate=drop, optimizer=opt)
    if m is None:
        print("ข้าม:", name, "(ยังไม่ได้ทำ build_mlp ในข้อ 4)")
        continue
    h = m.fit(X_train, y_train, epochs=EXP_EPOCHS, validation_split=0.1, verbose=0)
    results.append({
        "การทดลอง": name,
        "params": m.count_params(),
        "train_acc": max(h.history["accuracy"]),
        "best_val_acc": max(h.history["val_accuracy"]),
    })
    print(f"เสร็จ: {name}")

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False) if len(results_df) else "ยังไม่มีผลการทดลอง")
```

### ผลลัพธ์ที่คาดหวัง

ตัวอย่างช่วงค่า (ค่าที่ได้จะต่างกันเล็กน้อยในแต่ละครั้ง) — คอลัมน์ `params` เป็นค่าแน่นอน

| การทดลอง | params | best_val_acc (ประมาณ) |
|---|---|---|
| baseline 300-100, adam | 266,610 | 0.87–0.89 |
| เล็ก 64 | 50,890 | 0.86–0.88 |
| ใหญ่ 512-256 | 535,818 | 0.87–0.89 |
| Dropout 0.3 | 266,610 | 0.86–0.88 |
| SGD | 266,610 | 0.84–0.87 |
| RMSprop | 266,610 | 0.86–0.88 |

### คำถามท้ายข้อ

1. การเพิ่มจำนวน hidden units จาก 64 เป็น 512-256 (พารามิเตอร์เพิ่มกว่า 10 เท่า) ทำให้ val accuracy ดีขึ้นคุ้มค่าหรือไม่
2. ทำไม SGD (learning rate 0.01) จึงได้ผลต่ำกว่า Adam เมื่อฝึกเพียง 5 epoch ถ้าฝึกนานขึ้นผลจะเป็นอย่างไร
3. ในการทดลอง Dropout 0.3 train_acc ต่ำกว่า val_acc หรือไม่ อธิบายสาเหตุ

## ข้อที่ 6: ทางเลือกที่ไม่ต้องใช้ GPU — `MLPClassifier` ของ scikit-learn

### โจทย์

ฝึก `MLPClassifier` บนข้อมูลฝึก 20,000 ภาพแรก (เพื่อให้รันบน CPU ได้ภายในประมาณ 1 นาที) ด้วย hidden layer 128 neuron, Adam และ early stopping แล้วเปรียบเทียบกับผลของ Keras ในข้อ 4

### โค้ดเริ่มต้น

```python
from sklearn.neural_network import MLPClassifier

N = 20000
X_tr_flat = X_train[:N].reshape(N, -1)            # (20000, 784)
X_te_flat = X_test.reshape(len(X_test), -1)       # (10000, 784)

mlp = MLPClassifier(hidden_layer_sizes=(128,),    # TODO: ลองเปลี่ยนเป็น (300, 100)
                    activation="relu", solver="adam",
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, max_iter=30, random_state=42)
mlp.fit(X_tr_flat, y_train[:N])

n_params = sum(W.size for W in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
print("จำนวน epoch ที่ฝึก:", mlp.n_iter_)
print("จำนวนพารามิเตอร์:", f"{n_params:,}")
print("best validation accuracy:", round(mlp.best_validation_score_, 4))
print("test accuracy:", round(mlp.score(X_te_flat, y_test), 4))
```

### ผลลัพธ์ที่คาดหวัง

```output
ConvergenceWarning: Stochastic Optimizer: Maximum iterations (30) reached and the optimization hasn't converged yet.
จำนวน epoch ที่ฝึก: 30
จำนวนพารามิเตอร์: 101,770
best validation accuracy: 0.8665
test accuracy: 0.8668
```

### คำถามท้ายข้อ

1. จำนวนพารามิเตอร์ 101,770 ตรงกับสถาปัตยกรรมใดในข้อ 2 และ `ConvergenceWarning` หมายความว่าอะไร (เป็น error หรือไม่)
2. เปรียบเทียบ test accuracy กับ Keras ในข้อ 4 ปัจจัยใดทำให้ต่างกัน (จำนวนข้อมูลฝึก ขนาดโมเดล จำนวน epoch)
3. ข้อดีและข้อจำกัดของ `MLPClassifier` เทียบกับ Keras มีอะไรบ้าง

## ข้อที่ 7: วาดและอธิบายสถาปัตยกรรมเครือข่าย

### โจทย์

ตอบในเซลล์ markdown (วาดแผนภาพด้วยมือแล้วแนบรูป หรือเขียน mermaid / ตารางก็ได้) และใช้เซลล์โค้ดด้านล่างตรวจจำนวนพารามิเตอร์

1. **Breast cancer** (30 feature, ทำนาย มะเร็ง/ไม่เป็น): ออกแบบ MLP ที่มี hidden layer 2 ชั้น ระบุจำนวน neuron, activation ของทุกชั้น, loss, optimizer และจำนวนพารามิเตอร์
2. **California housing** (8 feature, ทำนายราคาบ้าน): ระบุ output layer และ loss ที่เหมาะสม และอธิบายว่าทำไมต้องทำ feature scaling ก่อนป้อนเข้า neural network
3. วาดแผนภาพของโมเดลที่ดีที่สุดของคุณจากข้อ 5 แสดงชั้น จำนวน neuron, activation, Dropout และ Output Shape ของแต่ละชั้น
4. อธิบาย 3 ขั้นตอนของ backpropagation ที่เกิดขึ้นใน **1 step** ของ `model.fit()` และบอกว่า 1 epoch ในข้อ 4 มีกี่ step

### โค้ดเริ่มต้น

```python
# TODO: ใส่จำนวน neuron ของแต่ละชั้น (รวม input และ output) ของโมเดลที่ออกแบบ
my_breast_cancer_mlp = [30, 1]       # TODO: เช่น [30, ?, ?, 1]
my_housing_mlp = [8, 1]              # TODO: เช่น [8, ?, 1]

for name, sizes in [("Breast cancer", my_breast_cancer_mlp), ("Housing", my_housing_mlp)]:
    print(f"{name}: {sizes} -> params = {mlp_params(sizes):,}")

steps_per_epoch = 0                  # TODO: ceil(จำนวนข้อมูลฝึกหลังแบ่ง validation / batch_size)
print("steps per epoch (ข้อ 4) =", steps_per_epoch)
```

### ผลลัพธ์ที่คาดหวัง

ตัวอย่างเมื่อเลือก 30-16-8-1 และ 8-30-1 (คำตอบของแต่ละคนต่างกันได้ แต่ต้องคำนวณถูก)

```output
Breast cancer: [30, 16, 8, 1] -> params = 641
Housing: [8, 30, 1] -> params = 301
steps per epoch (ข้อ 4) = 1688
```

### คำถามท้ายข้อ

1. ถ้าโจทย์ breast cancer เปลี่ยน output เป็น `Dense(2, activation="softmax")` ต้องเปลี่ยน loss เป็นอะไร ผลลัพธ์ต่างจาก `Dense(1, sigmoid)` อย่างไร
2. ถ้าเพิ่ม hidden layer เป็น 20 ชั้นโดยใช้ sigmoid จะเกิดปัญหาอะไร และควรแก้อย่างไร (ระบุอย่างน้อย 2 วิธี)

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | คะแนน | สิ่งที่ตรวจในใบงานนี้ |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | 2 | คำถามท้ายข้อ 1, 3 และ 7 (neuron, activation, backpropagation, vanishing gradients) |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | 2 | เลือก output layer / loss / optimizer ได้ถูกในข้อ 4 และ 7 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | 2 | ข้อ 4–5 ครบ build → compile → fit → plot → evaluate และตารางการทดลอง |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | 3 | forward pass ด้วยมือ/NumPy (ข้อ 1), จำนวนพารามิเตอร์ (ข้อ 2, 7), โค้ด Keras รันได้ |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | 1 | ตีความ learning curves และสรุปผลการทดลองในข้อ 5–6 |
| **รวม** | **10** | |

## สิ่งที่ต้องส่ง

1. ไฟล์ `lab-06_<รหัสนักศึกษา>.ipynb` ที่รันครบทุกเซลล์ (Runtime → Run all) และเห็นผลลัพธ์ทุกเซลล์
2. ภาพ/เซลล์ markdown แสดงการคำนวณ forward pass ด้วยมือของข้อ 1
3. กราฟ learning curves (ข้อ 4) และตารางผลการทดลองพร้อมข้อสังเกต (ข้อ 5)
4. แผนภาพสถาปัตยกรรมและคำตอบข้อ 7
5. คำตอบคำถามท้ายข้อทุกข้อในเซลล์ markdown ใต้แต่ละข้อ
