## ข้อมูลทั่วไปของใบงาน

| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Image Classification ด้วย CNN |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 10 |
| งานตาม มคอ.3 | งานปฏิบัติ 3: image classification |
| CLO/LLO | CLO3 (LLO3-1 อธิบายโครงสร้างและหลักการทำงานของ CNN, LLO3-2 พัฒนาแบบจำลอง Image Classification ตั้งแต่อ่านภาพ แปลงข้อมูล สร้างโมเดล ฝึก ทำนาย และประเมินผล) |
| เครื่องมือ | Google Colab (Runtime แบบ GPU), Python 3, Keras 3 / TensorFlow, NumPy, matplotlib, scikit-learn, Pillow |
| ข้อมูล | MNIST จาก `keras.datasets.mnist` (train 60,000 / test 10,000 ภาพ ขนาด 28×28) และภาพลายมือของนักศึกษาเอง 1–3 ภาพ |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ พร้อมคำตอบคำถามท้ายข้อ และตารางเปรียบเทียบ CNN กับ MLP |

### วัตถุประสงค์

1. เขียน convolution และ max pooling ด้วย NumPy เองได้ และอธิบายผลของ stride/padding ได้
2. คำนวณขนาด output และจำนวนพารามิเตอร์ของแต่ละชั้นใน CNN ได้ถูกต้องก่อนสร้างโมเดลจริง
3. เตรียมข้อมูลภาพสำหรับ CNN ได้ (reshape เพิ่ม channel, normalize, one-hot encoding)
4. สร้าง ฝึก และประเมิน CNN ด้วย Keras พร้อมอ่าน learning curve และ confusion matrix ได้
5. นำโมเดลไปทำนายภาพลายมือจริงที่อัปโหลดเอง และเปรียบเทียบ CNN กับ MLP จาก Lab 06 อย่างมีเหตุผล

> [!NOTE]
> ใบงานนี้ต่อยอดจากเนื้อหา [บทที่ 7](/module/07) ก่อนเริ่มให้เปิด GPU: เมนู **Runtime → Change runtime type → T4 GPU** แล้วรันเซลล์ตามลำดับจากบนลงล่าง ทุกเซลล์ที่มี `# TODO:` รันได้ตั้งแต่ต้น (ด้วยค่าชั่วคราว) แต่ผลลัพธ์จะถูกต้องเมื่อเติมโค้ดครบแล้วเท่านั้น เซลล์ที่ต้องฝึกโมเดลจะข้ามการฝึกให้อัตโนมัติจนกว่าข้อที่ 3 จะเสร็จ

## ข้อที่ 0: เตรียมสภาพแวดล้อม

```python
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import layers
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

keras.utils.set_random_seed(42)          # ให้ผลซ้ำได้ใกล้เคียงกันทุกครั้ง
print("Keras version:", keras.__version__)

import tensorflow as tf
print("GPU:", tf.config.list_physical_devices("GPU"))

# โหลด MNIST (ดาวน์โหลดอัตโนมัติครั้งแรก ~11 MB)
(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.mnist.load_data()
print("โหลดข้อมูลเรียบร้อย:", X_train_raw.shape, X_test_raw.shape)
```

ผลลัพธ์ที่ควรได้: Keras เวอร์ชัน 3.x, รายการ GPU ไม่ว่างเปล่า (ถ้าว่างแปลว่ายังไม่ได้เปิด GPU ยังทำต่อได้แต่ช้ากว่า) และ `โหลดข้อมูลเรียบร้อย: (60000, 28, 28) (10000, 28, 28)`

## ข้อที่ 1: Convolution และ Max Pooling ด้วย NumPy

### โจทย์

1. เติมฟังก์ชัน `conv2d` ให้คำนวณขนาด output ตามสูตร $\lfloor (n - k)/s \rfloor + 1$ (หลังเติม padding แล้ว) และคำนวณผลรวมของ `patch * kernel`
2. เติมฟังก์ชัน `max_pool2d` ให้เลือกค่าสูงสุดในแต่ละหน้าต่าง
3. ทดสอบกับภาพ 6×6 และ Sobel kernel แล้วตรวจกับ `scipy.signal.correlate2d`
4. นำ filter ตรวจจับขอบแนวตั้ง/แนวนอนไปใช้กับภาพตัวเลขจริงจาก MNIST แล้วแสดงผล

### โค้ดเริ่มต้น

```python
def conv2d(image, kernel, stride=1, padding=0):
    """convolution 2 มิติ (แบบที่ CNN ใช้ = cross-correlation) ช่องสีเดียว"""
    if padding > 0:
        image = np.pad(image, padding)          # เติม 0 รอบภาพ padding ชั้น
    kh, kw = kernel.shape
    out_h = 0  # TODO: (image.shape[0] - kh) // stride + 1
    out_w = 0  # TODO: (image.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = 0  # TODO: คูณ patch กับ kernel ทีละตัวแล้วบวกรวม (np.sum)
    return out


def max_pool2d(fmap, size=2, stride=2):
    """max pooling ขนาด size×size"""
    out_h = (fmap.shape[0] - size) // stride + 1
    out_w = (fmap.shape[1] - size) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = fmap[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = 0  # TODO: ค่าสูงสุดใน window
    return out
```

```python
from scipy.signal import correlate2d

image = np.array([[1, 2, 0, 3, 1, 0],
                  [0, 1, 4, 1, 2, 1],
                  [2, 0, 1, 5, 0, 2],
                  [1, 3, 2, 0, 1, 4],
                  [0, 2, 1, 2, 3, 1],
                  [4, 1, 0, 1, 2, 0]])
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

fm = conv2d(image, sobel_x)
print("feature map (valid):\n", fm)
relu = np.maximum(fm, 0)
print("หลัง ReLU:\n", relu)
print("หลัง MaxPool 2x2:\n", max_pool2d(relu))
print("shape เมื่อ padding=1:", conv2d(image, sobel_x, padding=1).shape)
print("shape เมื่อ stride=2 :", conv2d(image, sobel_x, stride=2).shape)
print("ตรงกับ scipy:", fm.shape == (4, 4) and np.allclose(fm, correlate2d(image, sobel_x, mode="valid")))
```

```python
# ใช้ filter กับภาพตัวเลขจริง (ภาพแรกของ train set)
digit = X_train_raw[0].astype("float32") / 255.0
v_edge = np.array([[1, 0, -1]] * 3)       # ขอบแนวตั้ง
h_edge = v_edge.T                         # ขอบแนวนอน

fm_v = conv2d(digit, v_edge, padding=1)
fm_h = conv2d(digit, h_edge, padding=1)

if fm_v.size > 0:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    panels = [(digit, "input"), (fm_v, "vertical edges"),
              (fm_h, "horizontal edges"), (max_pool2d(np.maximum(fm_v, 0)), "ReLU + MaxPool (vertical)")]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title}\n{img.shape}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("ยังเติม conv2d ไม่ครบ")
```

### ผลลัพธ์ที่คาดหวัง

```output
feature map (valid):
 [[-6. -6.  4.  6.]
 [-3. -7.  5.  2.]
 [-2.  1.  1. -4.]
 [ 1.  3. -5. -1.]]
หลัง ReLU:
 [[0. 0. 4. 6.]
 [0. 0. 5. 2.]
 [0. 1. 1. 0.]
 [1. 3. 0. 0.]]
หลัง MaxPool 2x2:
 [[0. 6.]
 [3. 1.]]
shape เมื่อ padding=1: (6, 6)
shape เมื่อ stride=2 : (2, 2)
ตรงกับ scipy: True
```

ภาพในเซลล์ที่สามมี 4 ช่อง: เลข 5 ขนาด (28, 28), feature map ขอบแนวตั้งและแนวนอนขนาด (28, 28) (เพราะ padding=1) และภาพหลัง ReLU + MaxPool ขนาด (14, 14)

### คำถามท้ายข้อ

1. คำนวณค่า `fm[0, 0] = -6` ด้วยมือ แสดงการคูณทีละตำแหน่ง
2. ทำไม padding=1 กับ kernel 3×3 จึงได้ขนาด output เท่า input และ stride=2 ให้ขนาด 2×2 ได้อย่างไร (แสดงด้วยสูตร)
3. ในภาพ vertical edges ส่วนใดของเลข 5 สว่าง (ค่าบวก/ลบมาก) และสอดคล้องกับทิศของ filter อย่างไร
4. ใน CNN จริง ค่าใน kernel มาจากไหน ต่างจาก Sobel kernel ในข้อนี้อย่างไร

## ข้อที่ 2: ตารางขนาด Output และจำนวนพารามิเตอร์

### โจทย์

1. เติม 3 ฟังก์ชัน: `conv_output_size`, `conv_params`, `dense_params`
2. **ก่อนรันเซลล์** ให้เขียนตารางด้วยมือลงในเซลล์ Markdown สำหรับสถาปัตยกรรม A และ B (output shape และ params ของทุกชั้น) แล้วจึงรันเพื่อตรวจคำตอบ
3. สถาปัตยกรรม A คือ CNN ที่จะสร้างจริงในข้อที่ 4 — ตัวเลขต้องตรงกับ `model.summary()`

### โค้ดเริ่มต้น

```python
def conv_output_size(n, k, s=1, p=0):
    """ขนาด output ของ Conv/Pool: floor((n + 2p - k) / s) + 1"""
    return 0  # TODO

def conv_params(k, c_in, filters):
    """พารามิเตอร์ของ Conv2D (kernel k×k): (k*k*c_in + 1) * filters"""
    return 0  # TODO

def dense_params(n_in, units):
    """พารามิเตอร์ของ Dense: weight + bias"""
    return 0  # TODO


def show_table(name, input_shape, layer_list):
    """พิมพ์ตาราง output shape และ params ของแต่ละชั้น (ให้มาแล้ว ไม่ต้องแก้)"""
    h, w, c = input_shape
    total = 0
    print(f"=== {name} | input {h}x{w}x{c}")
    print(f"{'layer':26s}{'output shape':>16s}{'params':>10s}")
    for spec in layer_list:
        kind = spec[0]
        if kind == "conv":
            _, filters, k, s, pad = spec
            p = (k - 1) // 2 if pad == "same" else 0
            h, w = conv_output_size(h, k, s, p), conv_output_size(w, k, s, p)
            params = conv_params(k, c, filters)
            c = filters
            label = f"Conv2D({filters}, {k}, s={s}, {pad})"
        elif kind == "pool":
            _, size = spec
            h, w = conv_output_size(h, size, size), conv_output_size(w, size, size)
            params = 0
            label = f"MaxPooling2D({size})"
        elif kind == "flatten":
            c, h, w = h * w * c, 1, 1
            params = 0
            label = "Flatten"
        elif kind == "dense":
            _, units = spec
            params = dense_params(h * w * c, units)
            h, w, c = 1, 1, units
            label = f"Dense({units})"
        total += params
        shape = f"{h}x{w}x{c}" if kind in ("conv", "pool") else f"{c}"
        print(f"{label:26s}{shape:>16s}{params:>10,}")
    print(f"{'Total params':26s}{'':>16s}{total:>10,}")


show_table("A: CNN ที่จะสร้างในข้อที่ 4", (28, 28, 1),
           [("conv", 32, 3, 1, "valid"), ("pool", 2),
            ("conv", 64, 3, 1, "valid"), ("pool", 2),
            ("flatten",), ("dense", 64), ("dense", 10)])
show_table("B: CNN สำหรับภาพสี", (64, 64, 3),
           [("conv", 16, 5, 1, "valid"), ("pool", 2),
            ("conv", 32, 3, 1, "same"), ("pool", 2),
            ("flatten",), ("dense", 10)])
```

### ผลลัพธ์ที่คาดหวัง

```output
=== A: CNN ที่จะสร้างในข้อที่ 4 | input 28x28x1
layer                         output shape    params
Conv2D(32, 3, s=1, valid)         26x26x32       320
MaxPooling2D(2)                   13x13x32         0
Conv2D(64, 3, s=1, valid)         11x11x64    18,496
MaxPooling2D(2)                     5x5x64         0
Flatten                               1600         0
Dense(64)                               64   102,464
Dense(10)                               10       650
Total params                                 121,930
=== B: CNN สำหรับภาพสี | input 64x64x3
layer                         output shape    params
Conv2D(16, 5, s=1, valid)         60x60x16     1,216
MaxPooling2D(2)                   30x30x16         0
Conv2D(32, 3, s=1, same)          30x30x32     4,640
MaxPooling2D(2)                   15x15x32         0
Flatten                               7200         0
Dense(10)                               10    72,010
Total params                                  77,866
```

### คำถามท้ายข้อ

1. ในสถาปัตยกรรม A ชั้นใดมีพารามิเตอร์มากที่สุด คิดเป็นกี่เปอร์เซ็นต์ของทั้งหมด และสื่ออะไรเกี่ยวกับ Conv layer กับ Dense layer
2. ถ้าเปลี่ยน input ของ A จาก 28×28×1 เป็นภาพสี 28×28×3 พารามิเตอร์ของ Conv2D ชั้นแรกและชั้นที่สองเปลี่ยนเป็นเท่าใด (ชั้นที่สองเปลี่ยนหรือไม่ เพราะอะไร)
3. ใน B ถ้าใช้ `GlobalAveragePooling2D()` แทน `Flatten` จำนวนพารามิเตอร์ของ `Dense(10)` จะเหลือเท่าใด

## ข้อที่ 3: โหลดและเตรียมข้อมูลภาพ

### โจทย์

1. ตรวจ shape, dtype และช่วงค่าของข้อมูลดิบ แล้วดูจำนวนภาพต่อคลาส
2. **reshape** ภาพเป็น `(N, 28, 28, 1)` แปลงเป็น `float32` และ **normalize** ด้วยการหาร 255
3. ทำ **one-hot encoding** ให้ label ด้วย `keras.utils.to_categorical`
4. แสดงตัวอย่างภาพ 10 ภาพพร้อม label

### โค้ดเริ่มต้น

```python
print("raw   :", X_train_raw.shape, X_test_raw.shape, X_train_raw.dtype,
      X_train_raw.min(), X_train_raw.max())
print("จำนวนภาพต่อคลาส (train):", np.bincount(y_train))

X_train = X_train_raw   # TODO: reshape เป็น (-1, 28, 28, 1), astype("float32") แล้วหาร 255.0
X_test = X_test_raw     # TODO: ทำแบบเดียวกับ X_train
y_train_oh = y_train    # TODO: keras.utils.to_categorical(y_train, 10)
y_test_oh = y_test      # TODO: ทำแบบเดียวกัน


def data_ready():
    """ตรวจว่าข้อมูลพร้อมเข้า CNN แล้วหรือยัง (ให้มาแล้ว)"""
    return (X_train.ndim == 4 and X_train.dtype == np.float32
            and X_train.max() <= 1.0 and y_train_oh.ndim == 2)


print("X_train:", X_train.shape, X_train.dtype, X_train.min(), X_train.max())
print("X_test :", X_test.shape)
print("y_train[0] =", y_train[0], "-> one-hot", y_train_oh[0])
print("y_train_oh:", y_train_oh.shape, " y_test_oh:", y_test_oh.shape)
print("ข้อมูลพร้อมฝึก:", data_ready())

fig, axes = plt.subplots(1, 10, figsize=(14, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_train_raw[i], cmap="gray")
    ax.set_title(f"label {y_train[i]}")
    ax.axis("off")
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
raw   : (60000, 28, 28) (10000, 28, 28) uint8 0 255
จำนวนภาพต่อคลาส (train): [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
X_train: (60000, 28, 28, 1) float32 0.0 1.0
X_test : (10000, 28, 28, 1)
y_train[0] = 5 -> one-hot [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
y_train_oh: (60000, 10)  y_test_oh: (10000, 10)
ข้อมูลพร้อมฝึก: True
```

ภาพตัวอย่าง 10 ภาพแรกมี label 5 0 4 1 9 2 1 3 1 4

### คำถามท้ายข้อ

1. ทำไม Conv2D ต้องการ input 4 มิติ `(N, H, W, C)` ทั้งที่ภาพ MNIST เป็นภาพเทา
2. ข้อมูลชุดนี้สมดุลหรือไม่ (ดูจาก `bincount`) และมีผลต่อการเลือก metric อย่างไร
3. ถ้าไม่ทำ one-hot encoding แต่ใช้ `y_train` ตรง ๆ ต้องเปลี่ยน loss ในข้อที่ 4 เป็นอะไร

## ข้อที่ 4: สร้างและฝึก CNN

### โจทย์

1. เติม layer ใน `keras.Sequential` ให้ตรงกับสถาปัตยกรรม A ในข้อที่ 2: Conv2D(32, 3, relu) → MaxPooling2D(2) → Conv2D(64, 3, relu) → MaxPooling2D(2) → Flatten → Dense(64, relu) → Dense(10, softmax)
2. ตรวจ `cnn.summary()` ว่าตรงกับตารางที่คำนวณด้วยมือ (Total params 121,930)
3. เลือก loss ให้ถูกกับ label แบบ one-hot แล้วฝึก 5 epochs, `batch_size=128`, `validation_split=0.1`
4. วาด learning curve (loss และ accuracy ของ train/validation)

### โค้ดเริ่มต้น

```python
cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    # TODO: layers.Conv2D(32, kernel_size=3, activation="relu"),
    # TODO: layers.MaxPooling2D(pool_size=2),
    # TODO: layers.Conv2D(64, kernel_size=3, activation="relu"),
    # TODO: layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    # TODO: layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="cnn")
cnn.summary()

cnn.compile(optimizer="adam",
            loss="mse",            # TODO: เปลี่ยนเป็น loss ที่เหมาะกับ label แบบ one-hot
            metrics=["accuracy"])
```

```python
history, cnn_time = None, None
if data_ready():
    t0 = time.time()
    history = cnn.fit(X_train, y_train_oh, batch_size=128, epochs=5,
                      validation_split=0.1, verbose=2)
    cnn_time = time.time() - t0
    print(f"เวลาฝึก CNN: {cnn_time:.1f} วินาที")
else:
    print("ยังเตรียมข้อมูลในข้อที่ 3 ไม่ครบ จึงยังไม่ฝึกโมเดล")
```

```python
def plot_history(history, title):
    """วาด learning curve จาก history ของ model.fit"""
    h = history.history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(h["loss"], "o-", label="train")
    ax1.plot(h["val_loss"], "o-", label="validation")
    ax1.set_title(f"{title} - loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(h["accuracy"], "o-", label="train")
    ax2.plot(h["val_accuracy"], "o-", label="validation")
    ax2.set_title(f"{title} - accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout()
    plt.show()


if history is not None:
    plot_history(history, "CNN")
```

### ผลลัพธ์ที่คาดหวัง

`cnn.summary()` (ชื่อ layer อาจมีเลขต่อท้ายต่างไป แต่ shape และ params ต้องตรง)

```output
Model: "cnn"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │       102,464 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 121,930 (476.29 KB)
 Trainable params: 121,930 (476.29 KB)
 Non-trainable params: 0 (0.00 B)
```

ตัวอย่างผลลัพธ์การฝึก (ค่าที่ได้จะต่างกันเล็กน้อยในแต่ละครั้ง): แต่ละ epoch มี 422 steps, val_accuracy ของ epoch แรกราว 0.97–0.98 และ epoch ที่ 5 ราว 0.985–0.99 บน GPU ใช้เวลาไม่ถึง 1 นาที (บน CPU อาจหลายนาที) กราฟ loss ของ train และ validation ควรลดลงใกล้เคียงกัน

### คำถามท้ายข้อ

1. ตัวเลขใน `cnn.summary()` ตรงกับตารางในข้อที่ 2 ทุกชั้นหรือไม่ ถ้าไม่ตรงเป็นเพราะอะไร
2. ถ้ายังใช้ `loss="mse"` อยู่ ผลการฝึกต่างจาก `categorical_crossentropy` อย่างไร (ลองทั้งสองแบบ) และทำไม cross-entropy จึงเหมาะกับ softmax มากกว่า
3. จาก learning curve มีสัญญาณของ overfitting หรือไม่ อธิบายจากกราฟ
4. ทำไม 1 epoch จึงมี 422 steps

## ข้อที่ 5: ประเมินผลด้วย Confusion Matrix และดูภาพที่ทายผิด

### โจทย์

1. ประเมิน test accuracy ด้วย `cnn.evaluate`
2. แปลงความน่าจะเป็นจาก `cnn.predict` เป็นคลาสด้วย `argmax`
3. สร้าง confusion matrix, classification report และหา 5 คู่ที่สับสนบ่อยที่สุด
4. แสดงภาพที่โมเดลทายผิด 15 ภาพ พร้อมคลาสจริง คลาสที่ทาย และความมั่นใจ

### โค้ดเริ่มต้น

```python
cnn_acc = None
if data_ready():
    test_loss, cnn_acc = cnn.evaluate(X_test, y_test_oh, verbose=0)
    print(f"Test loss = {test_loss:.4f}   Test accuracy = {cnn_acc:.4f}")

    y_prob = cnn.predict(X_test, verbose=0)               # (10000, 10)
    y_pred = np.zeros(len(y_test), dtype=int)             # TODO: np.argmax(y_prob, axis=1)
    print("ความน่าจะเป็นของภาพแรก:", y_prob[0].round(3), "-> ทาย", y_pred[0], "เฉลย", y_test[0])

    cm = confusion_matrix(y_test, y_pred, labels=range(10))
    ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d")
    plt.title("CNN on MNIST - confusion matrix")
    plt.show()
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # 5 คู่ (จริง -> ทาย) ที่ผิดบ่อยที่สุด
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    top = np.argsort(cm_off.ravel())[::-1][:5]
    for t, p in zip(*np.unravel_index(top, cm.shape)):
        print(f"จริง {t} -> ทาย {p}: {cm_off[t, p]} ภาพ")
else:
    print("ยังเตรียมข้อมูลในข้อที่ 3 ไม่ครบ")
```

```python
if cnn_acc is not None:
    wrong = np.where(y_pred != y_test)[0]
    print("จำนวนภาพที่ทายผิด:", len(wrong), "จาก", len(y_test))
    fig, axes = plt.subplots(3, 5, figsize=(11, 7))
    for ax, i in zip(axes.flat, wrong[:15]):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        ax.set_title(f"true {y_test[i]} / pred {y_pred[i]}\nconf {y_prob[i].max():.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
```

### ผลลัพธ์ที่คาดหวัง

ตัวอย่างผลลัพธ์ (ค่าที่ได้จะต่างกันเล็กน้อยในแต่ละครั้ง)

- Test accuracy ราว 0.985–0.99 จึงมีภาพทายผิดประมาณ 100–150 ภาพจาก 10,000
- ภาพแรกของ test set (เลข 7) ได้ความน่าจะเป็นของคลาส 7 ใกล้ 1.0
- confusion matrix มีค่าบนเส้นทแยงราว 970–1,130 ต่อคลาส คู่ที่ผิดบ่อยมักเป็นตัวเลขรูปทรงคล้ายกัน เช่น 4→9, 9→4, 7→2, 3→5, 5→3, 6→0
- precision/recall ของทุกคลาสอยู่ราว 0.97–1.00

### คำถามท้ายข้อ

1. คู่ที่สับสนบ่อยที่สุด 3 อันดับของคุณคืออะไร อธิบายจากรูปทรงของตัวเลขว่าทำไมจึงสับสน
2. ดูภาพที่ทายผิด 15 ภาพ มีกี่ภาพที่คุณเองก็ไม่แน่ใจว่าเป็นเลขอะไร สิ่งนี้บอกอะไรเกี่ยวกับขีดจำกัดของ accuracy
3. ภาพที่ทายผิดแต่โมเดล "มั่นใจสูง" (conf > 0.9) อันตรายกว่าภาพที่ผิดแบบมั่นใจต่ำอย่างไร ในงานจริง (เช่น อ่านเลขบัญชีจากเช็ค) ควรจัดการอย่างไร

## ข้อที่ 6: ทำนายภาพลายมือของคุณเอง

### โจทย์

1. เขียนตัวเลข 0–9 ด้วยปากกาเมจิกสีเข้มเส้นหนาบนกระดาษขาว ถ่ายภาพให้ตัวเลขอยู่กลางภาพ แล้วอัปโหลดเข้า Colab
2. เติมฟังก์ชัน `preprocess_digit` ให้ภาพผ่านขั้นตอนเดียวกับข้อมูลฝึก: grayscale → **กลับสี** (MNIST พื้นดำตัวเลขขาว) → จัดตัวเลขให้อยู่กลาง → resize 28×28 → ÷255 → reshape `(1, 28, 28, 1)`
3. ทำนายและแสดงภาพต้นฉบับ ภาพหลังเตรียม และกราฟความน่าจะเป็นทั้ง 10 คลาส
4. ทดลองปิดการจัดกลาง (`center=False`) หรือไม่กลับสี แล้วสังเกตผล

### โค้ดเริ่มต้น

```python
def center_digit(img, box=20, size=28):
    """(ให้มาแล้ว) ตัดเฉพาะตัวเลข ย่อให้พอดีกรอบ 20×20 แล้ววางกลางภาพ 28×28 แบบ MNIST
    รับภาพโหมด L ที่เป็นพื้นดำ ตัวเลขขาว"""
    img = ImageOps.autocontrast(img)
    arr = np.array(img)
    arr = np.where(arr > 80, arr, 0).astype(np.uint8)       # ตัดเงา/พื้นหลังจาง ๆ ทิ้ง
    img = Image.fromarray(arr)
    bbox = img.getbbox()                                      # กรอบของพิกเซลที่ไม่เป็น 0
    if bbox is None:
        return img.resize((size, size))
    img = img.crop(bbox)
    w, h = img.size
    scale = box / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    # เลื่อนให้จุดศูนย์กลางมวลอยู่กลางภาพ (MNIST จัดแบบนี้)
    arr = np.array(canvas).astype("float32")
    rows, cols = np.indices(arr.shape)
    cy, cx = (rows * arr).sum() / arr.sum(), (cols * arr).sum() / arr.sum()
    arr = np.roll(arr, (round(size / 2 - cy), round(size / 2 - cx)), axis=(0, 1))
    return Image.fromarray(arr.astype(np.uint8))


def preprocess_digit(img, center=True):
    """แปลงภาพ PIL (พื้นขาว หมึกดำ) -> array (1, 28, 28, 1) ค่า 0–1 แบบเดียวกับ MNIST"""
    img = img.convert("L")                                  # grayscale (ให้มาแล้ว)
    img = img                                               # TODO: กลับสีด้วย ImageOps.invert(img)
    if center:
        img = center_digit(img)
    img = img                                               # TODO: img.resize((28, 28))
    x = np.zeros((1, 28, 28, 1), dtype="float32")           # TODO: np.array(img) -> float32 -> /255.0 -> reshape(1, 28, 28, 1)
    return x
```

```python
try:
    from google.colab import files
    uploaded = files.upload()                    # เลือกไฟล์ภาพ .jpg/.png
    filename = next(iter(uploaded))
    user_img = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded[filename])))  # แก้ภาพมือถือที่หมุน
except (ImportError, StopIteration):
    print("ไม่ได้อัปโหลดภาพ — ใช้ภาพแรกของ test set (กลับสีเป็นพื้นขาวหมึกดำ) แทน")
    user_img = Image.fromarray(255 - X_test_raw[0])

x = preprocess_digit(user_img)
prob = cnn.predict(x, verbose=0)[0]
pred = int(np.argmax(prob))
print("input shape:", x.shape, " min/max:", x.min(), x.max())
print(f"โมเดลทายว่าเป็นเลข {pred} (ความมั่นใจ {prob[pred]:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
axes[0].imshow(user_img, cmap="gray"); axes[0].set_title("original"); axes[0].axis("off")
axes[1].imshow(x.squeeze(), cmap="gray"); axes[1].set_title("preprocessed 28x28"); axes[1].axis("off")
axes[2].bar(range(10), prob); axes[2].set_xticks(range(10))
axes[2].set_title(f"prediction = {pred}"); axes[2].set_xlabel("class")
plt.tight_layout()
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

เมื่อไม่ได้อัปโหลดภาพ (ใช้ภาพสำรองจาก test set ซึ่งคือเลข 7) และเติม TODO ครบ

```output
ไม่ได้อัปโหลดภาพ — ใช้ภาพแรกของ test set (กลับสีเป็นพื้นขาวหมึกดำ) แทน
input shape: (1, 28, 28, 1)  min/max: 0.0 1.0
```

ตามด้วย "โมเดลทายว่าเป็นเลข 7" ความมั่นใจใกล้ 1.0 สำหรับภาพลายมือที่ถ่ายเอง ภาพหลังเตรียมต้องเป็น **พื้นดำ ตัวเลขขาว อยู่กลางภาพ** หน้าตาคล้ายภาพใน MNIST ถ้าภาพหลังเตรียมยังเป็นพื้นขาว แปลว่ายังไม่ได้กลับสี

### คำถามท้ายข้อ

1. แสดงผลการทำนายภาพลายมือของคุณอย่างน้อย 3 ภาพ ทายถูกกี่ภาพ
2. ถ้าไม่กลับสี (ข้าม `ImageOps.invert`) หรือใช้ `center=False` ผลการทำนายเปลี่ยนไปอย่างไร เพราะอะไร
3. อธิบายแนวคิด **"train/serve skew"** (ข้อมูลตอนใช้งานจริงต่างจากตอนฝึก) จากประสบการณ์ในข้อนี้

## ข้อที่ 7: เปรียบเทียบ CNN กับ MLP (Lab 06)

### โจทย์

1. สร้าง MLP แบบเดียวกับ Lab 06: Flatten → Dense(128, relu) → Dense(10, softmax) แล้วฝึกด้วยข้อมูลและจำนวน epoch เท่ากับ CNN
2. สร้างตารางเปรียบเทียบ: จำนวนพารามิเตอร์ เวลาฝึก test accuracy และจำนวนภาพที่ทายผิด
3. ทดสอบความทนทานต่อการเลื่อนภาพ: เลื่อนภาพ test ไปทางขวา 3 พิกเซล แล้ววัด accuracy ของทั้งสองโมเดลใหม่
4. เขียนรายงานสรุป 5–8 บรรทัด

### โค้ดเริ่มต้น

```python
mlp = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    # TODO: layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
], name="mlp")
mlp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("MLP params:", mlp.count_params(), " CNN params:", cnn.count_params())

mlp_history, mlp_time, mlp_acc = None, None, None
if data_ready():
    t0 = time.time()
    mlp_history = mlp.fit(X_train, y_train_oh, batch_size=128, epochs=5,
                          validation_split=0.1, verbose=2)
    mlp_time = time.time() - t0
    _, mlp_acc = mlp.evaluate(X_test, y_test_oh, verbose=0)
    plot_history(mlp_history, "MLP")
```

```python
if data_ready() and cnn_acc is not None and mlp_acc is not None:
    # เลื่อนภาพ test ไปทางขวา 3 พิกเซล (axis=2 คือแกนความกว้าง)
    X_test_shift = np.roll(X_test, shift=3, axis=2)
    _, cnn_acc_shift = cnn.evaluate(X_test_shift, y_test_oh, verbose=0)
    _, mlp_acc_shift = mlp.evaluate(X_test_shift, y_test_oh, verbose=0)

    results = pd.DataFrame({
        "model": ["MLP", "CNN"],
        "params": [mlp.count_params(), cnn.count_params()],
        "train time (s)": [round(mlp_time, 1), round(cnn_time, 1)],
        "test accuracy": [round(mlp_acc, 4), round(cnn_acc, 4)],
        "errors (of 10,000)": [round((1 - mlp_acc) * 10000), round((1 - cnn_acc) * 10000)],
        "accuracy (shift 3 px)": [round(mlp_acc_shift, 4), round(cnn_acc_shift, 4)],
    })
    print(results.to_string(index=False))
else:
    print("ต้องฝึกทั้ง CNN และ MLP ให้เสร็จก่อน")
```

### ผลลัพธ์ที่คาดหวัง

เมื่อเติม TODO ครบ บรรทัดแรกต้องได้ค่าที่แน่นอนดังนี้ ($784\cdot128+128 = 100{,}480$ และ $128\cdot10+10 = 1{,}290$)

```output
MLP params: 101770  CNN params: 121930
```

ตัวอย่างผลลัพธ์ตารางเปรียบเทียบ (ค่าที่ได้จะต่างกันเล็กน้อยในแต่ละครั้ง และเวลาขึ้นกับ GPU/CPU)

| model | params | test accuracy | errors (of 10,000) | accuracy (shift 3 px) |
|---|---|---|---|---|
| MLP | 101,770 | ≈ 0.97–0.98 | ≈ 200–300 | มักลดลงอย่างชัดเจน |
| CNN | 121,930 | ≈ 0.985–0.99 | ≈ 100–150 | มักลดลงน้อยกว่า MLP |

### คำถามท้ายข้อ

1. CNN และ MLP มีจำนวนพารามิเตอร์ใกล้เคียงกัน แต่ CNN ทายผิดน้อยกว่าประมาณกี่เท่า อธิบายด้วยแนวคิด local connectivity และ weight sharing
2. เมื่อเลื่อนภาพ 3 พิกเซล โมเดลใด accuracy ลดลงมากกว่า เชื่อมโยงกับคุณสมบัติ translation invariance ของ Conv + Pooling
3. CNN ใช้เวลาฝึกต่อ epoch มากกว่า MLP หรือไม่ เพราะอะไร (คิดจากจำนวนการคูณ ไม่ใช่จำนวนพารามิเตอร์)
4. **รายงานสรุป (5–8 บรรทัด):** สรุปผลการทดลองทั้งหมด เลือกโมเดลที่จะนำไปใช้อ่านตัวเลขลายมือจริงพร้อมเหตุผล และระบุข้อจำกัดอย่างน้อย 2 ข้อ (เช่น ภาพจริงต่างจาก MNIST, ตัวเลขหลายหลักในภาพเดียว)

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | รายละเอียดในใบงานนี้ | คะแนน |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | อธิบาย convolution, filter/feature map, stride/padding, pooling, one-hot encoding และเหตุผลที่ CNN เหมาะกับภาพได้ถูกต้อง | 2 |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | เลือก loss ให้ถูกกับรูปแบบ label, เลือก preprocessing ให้ภาพจริงตรงกับข้อมูลฝึก และเปรียบเทียบ CNN กับ MLP อย่างมีเหตุผล | 2 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | ทำครบ อ่านภาพ → แปลงข้อมูล → สร้างโมเดล → ฝึก → ทำนาย → ประเมินผล ทั้งกับ test set และภาพของตนเอง | 2 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | เติม TODO ครบ conv2d/max_pool2d ถูกต้อง ตารางขนาดและพารามิเตอร์ตรงกับ `summary()` confusion matrix และตารางเปรียบเทียบถูกต้อง | 3 |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | ตีความ learning curve, คู่ที่สับสน, ภาพที่ทายผิด และเขียนรายงานสรุปพร้อมข้อจำกัด | 1 |
| **รวม** | | **10** |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab07_รหัสนักศึกษา.ipynb` ที่เติม TODO ครบ และรันทุกเซลล์จนเห็นผลลัพธ์ (Runtime → Run all) รวมถึงตารางที่เขียนด้วยมือในข้อที่ 2
2. ภาพลายมือที่ใช้ทดสอบในข้อที่ 6 อย่างน้อย 3 ภาพ พร้อมผลการทำนาย
3. คำตอบคำถามท้ายข้อทุกข้อ เขียนในเซลล์ Markdown ใต้ข้อนั้น ๆ
4. รายงานสรุปเปรียบเทียบ CNN กับ MLP (ข้อที่ 7 คำถามที่ 4)
5. ส่งภายในวันก่อนเริ่มคาบเรียนสัปดาห์ที่ 11 ผ่านช่องทางที่ผู้สอนกำหนด
