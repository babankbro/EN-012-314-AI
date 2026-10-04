## ข้อมูลทั่วไปของใบงาน

| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 01: จับคู่ปัญหากับรูปแบบการเรียนรู้ |
| งานตาม มคอ.3 | แบบฝึกหัด 2: ประเภท ML และ knowledge representation |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 2 |
| CLO/LLO | CLO1 (LLO1-2 ประเภทของ ML, LLO1-3 การแทนความรู้ด้วยกฎและระบบฐานความรู้) |
| เครื่องมือ | Google Colab, Python, pandas, scikit-learn, matplotlib |
| ชุดข้อมูล | `pima_indians_diabetes_real.csv` (768 แถว, 8 feature, target `Outcome`) |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ พร้อมคำตอบคำถามท้ายข้อ |

### วัตถุประสงค์

1. วิเคราะห์ปัญหาจริงแล้วจับคู่กับประเภทการเรียนรู้ (supervised / unsupervised / semi-supervised / self-supervised / reinforcement) และชนิดของผลลัพธ์ได้
2. แทนความรู้ด้วยกฎ IF-THEN และเขียน inference engine แบบ forward chaining อย่างง่ายได้
3. สร้างตัวจำแนกแบบกฎ (rule-based classifier) จากความรู้เชิงโดเมน และเปรียบเทียบกับโมเดลที่เรียนรู้จากข้อมูล (Decision Tree)
4. สังเกตปรากฏการณ์ overfitting/underfitting และใช้ validation set กับ cross-validation เลือก hyperparameter โดยไม่แตะ test set

## ข้อที่ 0: เตรียมสภาพแวดล้อม

### โจทย์

นำเข้าไลบรารีและโหลดไฟล์ `pima_indians_diabetes_real.csv` (ดาวน์โหลดจากหน้าเว็บรายวิชา) — ใน Colab เมื่อรันเซลล์จะมีปุ่ม **Choose Files** ให้เลือกไฟล์ หรือถ้าอาจารย์ให้ลิงก์ raw URL ให้วางในตัวแปร `DATA_URL` แทน

### โค้ดเริ่มต้น

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

pd.set_option("display.max_columns", None)   # แสดงทุกคอลัมน์
pd.set_option("display.width", 200)

DATA_URL = ""   # (ไม่บังคับ) วาง raw URL ของไฟล์ CSV ที่นี่ เช่น "https://.../pima_indians_diabetes_real.csv"
FILE_NAME = "pima_indians_diabetes_real.csv"

if DATA_URL:
    df = pd.read_csv(DATA_URL)
else:
    try:
        from google.colab import files        # ทำงานเฉพาะใน Google Colab
        uploaded = files.upload()             # เลือกไฟล์ pima_indians_diabetes_real.csv
        FILE_NAME = next(iter(uploaded))
    except ImportError:
        print("ไม่ได้รันบน Colab — อ่านไฟล์จากโฟลเดอร์ปัจจุบัน")
    df = pd.read_csv(FILE_NAME)

print("ขนาดข้อมูล:", df.shape)
print(df.head())
```

### ผลลัพธ์ที่คาดหวัง

```output
ขนาดข้อมูล: (768, 9)
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1
```

## ข้อที่ 1: จับคู่ปัญหากับรูปแบบการเรียนรู้

### โจทย์

อ่านปัญหา 10 ข้อในตาราง แล้วระบุ (1) **ประเภทการเรียนรู้** และ (2) **ชนิดของงาน/ผลลัพธ์** ลงในตัวแปร `answers` พร้อมเขียนเหตุผลสั้น ๆ ในคำถามท้ายข้อ

| ข้อ | ปัญหา |
|---|---|
| 1 | คัดกรองอีเมลขยะ โดยใช้อีเมลเก่าที่ผู้ใช้กด "รายงานสแปม" ไว้แล้ว 100,000 ฉบับ |
| 2 | ทำนายราคาที่ดิน (บาท/ตร.ว.) ในจังหวัดกาฬสินธุ์ จากทำเล ขนาด และระยะห่างจากถนนใหญ่ |
| 3 | แบ่งกลุ่มลูกค้าร้านค้าออนไลน์ตามพฤติกรรมการซื้อ โดยยังไม่รู้ว่าควรมีกี่กลุ่มหรือกลุ่มอะไรบ้าง |
| 4 | สอนหุ่นยนต์ดูดฝุ่นให้หาเส้นทางทำความสะอาด โดยได้คะแนนเมื่อเก็บฝุ่นได้และถูกหักคะแนนเมื่อชนเฟอร์นิเจอร์ |
| 5 | จำแนกภาพใบข้าวเป็น "โรคไหม้ / ใบจุด / ปกติ" มีภาพ 50,000 ภาพ แต่ผู้เชี่ยวชาญติด label ได้เพียง 500 ภาพ |
| 6 | พยากรณ์ปริมาณน้ำฝนของวันพรุ่งนี้ (มิลลิเมตร) จากข้อมูลสถานีตรวจอากาศย้อนหลัง |
| 7 | ตรวจจับธุรกรรมบัตรเครดิตที่ผิดปกติ ทั้งที่แทบไม่มีตัวอย่างธุรกรรมทุจริตที่ติด label ไว้ |
| 8 | ลดข้อมูลการแสดงออกของยีน 20,000 มิติ ให้เหลือ 2 มิติเพื่อพล็อตดูโครงสร้างข้อมูล |
| 9 | คำนวณค่าธรรมเนียมการศึกษาตามระเบียบมหาวิทยาลัย (จำนวนหน่วยกิต × อัตราต่อหน่วยกิต + ค่าบำรุง) |
| 10 | ฝึกโมเดลภาษาไทยให้ทายคำถัดไปจากข้อความบนเว็บจำนวนมหาศาล โดยไม่มีคนติด label |

### โค้ดเริ่มต้น

```python
# ประเภทการเรียนรู้: "Supervised", "Unsupervised", "Semi-supervised", "Self-supervised",
#                   "Reinforcement", "Rule-based (ไม่ใช้ ML)"
# ชนิดของงาน:       "Classification", "Regression", "Clustering", "Dimensionality reduction",
#                   "Anomaly detection", "Policy learning", "Next-word prediction", "คำนวณตามกฎ"
answers = {
    1: ("", ""),   # TODO: เช่น ("Supervised", "Classification")
    2: ("", ""),   # TODO
    3: ("", ""),   # TODO
    4: ("", ""),   # TODO
    5: ("", ""),   # TODO
    6: ("", ""),   # TODO
    7: ("", ""),   # TODO
    8: ("", ""),   # TODO
    9: ("", ""),   # TODO
    10: ("", ""),  # TODO
}

answer_df = pd.DataFrame.from_dict(answers, orient="index",
                                   columns=["ประเภทการเรียนรู้", "ชนิดของงาน"])
print(answer_df)
```

### ผลลัพธ์ที่คาดหวัง

ตาราง 10 แถว 2 คอลัมน์ที่กรอกครบทุกช่อง (ตรวจคำตอบกับเฉลยด้านล่างหลังทำเสร็จ)

<details><summary>เฉลย</summary>

| ข้อ | ประเภทการเรียนรู้ | ชนิดของงาน | เหตุผล |
|---|---|---|---|
| 1 | Supervised | Classification (binary) | มี label spam/ไม่ spam ครบ และคำตอบเป็นหมวดหมู่ |
| 2 | Supervised | Regression | มีราคาซื้อขายจริงเป็น label และคำตอบเป็นตัวเลขต่อเนื่อง |
| 3 | Unsupervised | Clustering | ไม่มี label ของกลุ่ม ต้องให้อัลกอริทึมหากลุ่มเอง |
| 4 | Reinforcement | Policy learning | agent เรียนจาก reward/penalty ที่ได้จากการกระทำ ไม่มีคำตอบที่ถูกในแต่ละก้าว |
| 5 | Semi-supervised | Classification (3 คลาส) | มี label เพียง 1% ใช้ภาพที่ไม่มี label ช่วยเรียนได้ |
| 6 | Supervised | Regression | ปริมาณน้ำฝนเป็นตัวเลขต่อเนื่อง และมีค่าจริงย้อนหลังเป็น label |
| 7 | Unsupervised | Anomaly detection | เรียนรูปแบบธุรกรรม "ปกติ" แล้วแจ้งเตือนสิ่งที่ต่างออกไป |
| 8 | Unsupervised | Dimensionality reduction | ลดมิติเพื่อการแสดงผล ไม่มี label |
| 9 | Rule-based (ไม่ใช้ ML) | คำนวณตามกฎ | สูตรกำหนดชัดเจนในระเบียบ ต้องถูกต้อง 100% ไม่ต้องเรียนจากข้อมูล |
| 10 | Self-supervised | Next-word prediction | label (คำถัดไป) สร้างจากตัวข้อความเอง |

</details>

### คำถามท้ายข้อ

1. ข้อ 2 และ 6 ถ้าเปลี่ยนคำถามเป็น "ราคาที่ดินแปลงนี้ สูงกว่าราคาประเมินหรือไม่" และ "พรุ่งนี้ฝนตกหรือไม่" ชนิดของงานจะเปลี่ยนเป็นอะไร
2. ข้อ 7 ถ้าในอนาคตธนาคารรวบรวมตัวอย่างธุรกรรมทุจริตที่ยืนยันแล้วได้หลายพันรายการ ควรเปลี่ยนไปใช้การเรียนรู้แบบใด
3. ยกตัวอย่างปัญหาในชีวิตประจำวันหรือในมหาวิทยาลัยของนักศึกษาเองอีก 2 ปัญหา พร้อมระบุประเภทการเรียนรู้

## ข้อที่ 2: ฐานความรู้และ Forward Chaining

### โจทย์

ระบบผู้เชี่ยวชาญจำแนกสัตว์มีฐานความรู้ (knowledge base) เป็นกฎ IF-THEN 3 ข้อ และมี inference engine แบบ forward chaining ให้แล้ว

1. เพิ่มกฎอีก 2 ข้อลงใน `new_rules`
   - IF เป็นสัตว์กินเนื้อ AND มีลายสีดำบนพื้นเหลือง THEN เป็นเสือ
   - IF เป็นนก AND บินไม่ได้ AND ว่ายน้ำได้ THEN เป็นเพนกวิน
2. รันระบบกับสัตว์ 2 ตัว แล้วตรวจว่าระบบสรุปได้ถูกต้องหรือไม่

### โค้ดเริ่มต้น

```python
# Knowledge base: (ชุดเงื่อนไขที่ต้องเป็นจริงทั้งหมด, ข้อสรุป)
rules = [
    ({"มีขน", "ให้นมลูก"}, "เป็นสัตว์เลี้ยงลูกด้วยนม"),
    ({"มีขนนก", "วางไข่"}, "เป็นนก"),
    ({"เป็นสัตว์เลี้ยงลูกด้วยนม", "กินเนื้อ"}, "เป็นสัตว์กินเนื้อ"),
]

new_rules = [
    # TODO: เพิ่มกฎ "เป็นเสือ" และ "เป็นเพนกวิน" ในรูปแบบ ({เงื่อนไข1, เงื่อนไข2, ...}, "ข้อสรุป")
]
rules.extend(new_rules)

def forward_chaining(facts, rules, verbose=True):
    facts = set(facts)                          # working memory
    fired = True
    while fired:                                # วนจนไม่มีข้อเท็จจริงใหม่
        fired = False
        for conditions, conclusion in rules:
            if conditions <= facts and conclusion not in facts:
                facts.add(conclusion)
                if verbose:
                    print(f"  IF {' AND '.join(sorted(conditions))} THEN {conclusion}")
                fired = True
    return facts

animal_1 = {"มีขน", "ให้นมลูก", "กินเนื้อ", "มีลายสีดำบนพื้นเหลือง"}
animal_2 = {"มีขนนก", "วางไข่", "บินไม่ได้", "ว่ายน้ำได้"}

for name, facts in [("สัตว์ตัวที่ 1", animal_1), ("สัตว์ตัวที่ 2", animal_2)]:
    print(name)
    result = forward_chaining(facts, rules)
    print("  ข้อเท็จจริงใหม่:", sorted(result - facts))
```

### ผลลัพธ์ที่คาดหวัง

```output
สัตว์ตัวที่ 1
  IF มีขน AND ให้นมลูก THEN เป็นสัตว์เลี้ยงลูกด้วยนม
  IF กินเนื้อ AND เป็นสัตว์เลี้ยงลูกด้วยนม THEN เป็นสัตว์กินเนื้อ
  IF มีลายสีดำบนพื้นเหลือง AND เป็นสัตว์กินเนื้อ THEN เป็นเสือ
  ข้อเท็จจริงใหม่: ['เป็นสัตว์กินเนื้อ', 'เป็นสัตว์เลี้ยงลูกด้วยนม', 'เป็นเสือ']
สัตว์ตัวที่ 2
  IF มีขนนก AND วางไข่ THEN เป็นนก
  IF บินไม่ได้ AND ว่ายน้ำได้ AND เป็นนก THEN เป็นเพนกวิน
  ข้อเท็จจริงใหม่: ['เป็นนก', 'เป็นเพนกวิน']
```

> [!NOTE]
> ถ้ายังไม่ได้เพิ่มกฎ ระบบจะสรุปได้แค่ "เป็นสัตว์เลี้ยงลูกด้วยนม / เป็นสัตว์กินเนื้อ" และ "เป็นนก" เท่านั้น — นี่คือข้อจำกัดของระบบกฎ: **รู้เท่าที่มนุษย์เขียนให้**

### คำถามท้ายข้อ

1. เขียนกฎ "เป็นเสือ" ในรูปตรรกศาสตร์เชิงประพจน์ (ใช้สัญลักษณ์ $\land$ และ $\Rightarrow$) และอธิบายว่าการได้ข้อสรุป "เป็นเสือ" ใช้ Modus Ponens กี่ครั้ง
2. ถ้าต้องการถามระบบว่า "สัตว์ตัวนี้เป็นเพนกวินหรือไม่" โดยเริ่มจากเป้าหมาย ควรใช้ forward หรือ backward chaining เพราะอะไร
3. ถ้าป้อนข้อเท็จจริง {มีขน, ให้นมลูก, กินเนื้อ} (สิงโต) ระบบจะสรุปอะไรได้บ้าง ขาดอะไรไปจึงจำแนกเป็นสิงโตไม่ได้

## ข้อที่ 3: สำรวจข้อมูลเบาหวานและแบ่ง Train/Test

### โจทย์

1. ดูการกระจายของคลาส `Outcome` (0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน)
2. นับจำนวนค่า 0 ในคอลัมน์ที่ค่า 0 เป็นไปไม่ได้ทางการแพทย์ (ปัญหา **poor-quality data** จากบทเรียน)
3. แบ่งข้อมูล train 80% / test 20% แบบ stratified แล้วดูค่าเฉลี่ยของ `Glucose`, `BMI`, `Age` แยกตามคลาส **จาก training set เท่านั้น** (เพื่อใช้ออกแบบกฎในข้อถัดไป)

### โค้ดเริ่มต้น

```python
print(df["Outcome"].value_counts())

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
zero_counts = None   # TODO: นับจำนวนค่า 0 ในแต่ละคอลัมน์ของ cols  (คำใบ้: (df[cols] == 0).sum())
print(zero_counts)

X = df.drop(columns="Outcome")
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", X_train.shape, " test:", X_test.shape)

# ค่าเฉลี่ยแยกตามคลาส (ใช้เฉพาะ training set)
print(X_train.join(y_train).groupby("Outcome")[["Glucose", "BMI", "Age"]].mean().round(1))
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
         Glucose   BMI   Age
Outcome                     
0          109.9  30.2  31.3
1          141.6  35.2  37.3
```

### คำถามท้ายข้อ

1. ข้อมูลมีคลาส 0 มากกว่าคลาส 1 ประมาณกี่เท่า ทำไมจึงต้องใช้ `stratify=y`
2. `Insulin` มีค่า 0 ถึง 374 แถว (เกือบครึ่ง) เป็นปัญหาแบบใดในหัวข้อ "ความท้าทายหลักของ ML" และควรจัดการอย่างไร (จะได้ทำจริงใน [บทที่ 2](/module/02))
3. ทำไมจึงควรดูค่าเฉลี่ยจาก training set เท่านั้น ไม่ใช่จากข้อมูลทั้งหมด

## ข้อที่ 4: ตัวจำแนกแบบกฎ (Rule-based Classifier)

### โจทย์

ทำตัวเป็น "ผู้เชี่ยวชาญ" เขียนฟังก์ชัน `rule_based_predict(row)` ที่คืนค่า 1 (เสี่ยงเป็นเบาหวาน) หรือ 0 ด้วยกฎ IF-THEN อย่างน้อย 2 กฎ โดยใช้ความรู้ทางการแพทย์ประกอบกับค่าเฉลี่ยจากข้อ 3 (ตัวอย่างความรู้: ระดับน้ำตาลหลังดื่มกลูโคส 2 ชั่วโมงตั้งแต่ 140 mg/dL ขึ้นไปถือว่าผิดปกติ, BMI ≥ 30 คือภาวะอ้วน, ความเสี่ยงเพิ่มตามอายุ) แล้ววัด accuracy บน training set และ test set

### โค้ดเริ่มต้น

```python
def rule_based_predict(row):
    # TODO: เขียนกฎ IF-THEN อย่างน้อย 2 กฎ เช่น
    # if row["Glucose"] > ...:
    #     return 1
    # if row["Glucose"] > ... and row["BMI"] > ... :
    #     return 1
    return 0   # ค่าเริ่มต้น: ทำนายว่าไม่เป็นเบาหวานเสมอ (baseline)

y_pred_rule_train = X_train.apply(rule_based_predict, axis=1)
y_pred_rule = X_test.apply(rule_based_predict, axis=1)
print(f"Rule-based  train accuracy = {accuracy_score(y_train, y_pred_rule_train):.3f}")
print(f"Rule-based  test  accuracy = {accuracy_score(y_test, y_pred_rule):.3f}")
```

### ผลลัพธ์ที่คาดหวัง

ก่อนเขียนกฎ (ทำนาย 0 เสมอ) จะได้ค่า **baseline** ซึ่งเท่ากับสัดส่วนคลาส 0

```output
Rule-based  train accuracy = 0.651
Rule-based  test  accuracy = 0.649
```

หลังเขียนกฎที่สมเหตุสมผล ควรได้ test accuracy ราว 0.70–0.78 (ขึ้นกับกฎของแต่ละคน) ตัวอย่างผลจากกฎของผู้สอน

```output
Rule-based  train accuracy = 0.777
Rule-based  test  accuracy = 0.747
```

<details><summary>ตัวอย่างกฎของผู้สอน</summary>

```text
IF Glucose > 140 THEN 1
ELSE IF Glucose > 110 AND BMI > 30 AND Age > 30 THEN 1
ELSE 0
```

</details>

### คำถามท้ายข้อ

1. เขียนกฎของนักศึกษาในรูป IF-THEN และอธิบายที่มาของค่า threshold แต่ละค่า
2. กฎของนักศึกษาดีกว่า baseline เท่าไร ถ้าทำนาย 0 เสมอได้ accuracy 0.649 อยู่แล้ว เหตุใด accuracy เพียงอย่างเดียวจึงอาจไม่พอสำหรับงานคัดกรองโรค
3. ถ้าปรับ threshold ไปเรื่อย ๆ จน test accuracy สูงที่สุด จะเกิดปัญหาอะไร (เชื่อมกับ data snooping)

## ข้อที่ 5: ให้เครื่องเรียนกฎเอง — Decision Tree

### โจทย์

1. สร้าง `DecisionTreeClassifier(max_depth=3, random_state=42)` แล้ว fit กับ training set
2. วัด accuracy บน train/test และเปรียบเทียบกับตัวจำแนกแบบกฎในข้อ 4
3. ใช้ `export_text` ดู "กฎ" ที่ต้นไม้เรียนรู้ได้

### โค้ดเริ่มต้น

```python
tree = None   # TODO: DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)

if tree is not None:
    print(f"Decision Tree train accuracy = {tree.score(X_train, y_train):.3f}")
    print(f"Decision Tree test  accuracy = {tree.score(X_test, y_test):.3f}")
    print(export_text(tree, feature_names=list(X.columns)))
```

### ผลลัพธ์ที่คาดหวัง

```output
Decision Tree train accuracy = 0.767
Decision Tree test  accuracy = 0.695
|--- Glucose <= 154.50
|   |--- BMI <= 27.35
|   |   |--- Glucose <= 151.50
|   |   |   |--- class: 0
|   |   |--- Glucose >  151.50
|   |   |   |--- class: 1
|   |--- BMI >  27.35
|   |   |--- Age <= 30.50
|   |   |   |--- class: 0
|   |   |--- Age >  30.50
|   |   |   |--- class: 0
|--- Glucose >  154.50
|   |--- BMI <= 23.10
|   |   |--- class: 0
|   |--- BMI >  23.10
|   |   |--- Insulin <= 544.00
|   |   |   |--- class: 1
|   |   |--- Insulin >  544.00
|   |   |   |--- class: 0
```

### คำถามท้ายข้อ

1. ต้นไม้เลือก feature ใดเป็นจุดแบ่งแรก ตรงกับความรู้ทางการแพทย์และกฎที่นักศึกษาเขียนหรือไม่
2. แปลงกิ่งที่ทำนาย `class: 1` ให้เป็นกฎ IF-THEN ภาษาไทย
3. ในการแบ่งข้อมูลครั้งนี้ กฎของผู้เชี่ยวชาญ (ข้อ 4) ได้ test accuracy ใกล้เคียงหรือสูงกว่าต้นไม้ max_depth=3 นักศึกษาคิดว่าเพราะอะไร และสรุปข้อดี-ข้อเสียของระบบกฎ vs โมเดลที่เรียนรู้จากข้อมูล (อย่างน้อยด้านละ 2 ข้อ)

## ข้อที่ 6: สังเกต Overfitting ด้วย Validation Set

### โจทย์

แบ่ง training set ออกอีกครั้งเป็น `X_tr` (75%) และ `X_val` (25%) แล้วฝึก Decision Tree ที่ `max_depth` = 1 ถึง 15 บันทึก accuracy บน train และ validation จากนั้นพล็อตกราฟ (ห้ามใช้ test set ในข้อนี้)

### โค้ดเริ่มต้น

```python
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

depths = range(1, 16)
train_acc, val_acc = [], []
for d in depths:
    model = None   # TODO: DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
    if model is None:
        continue
    train_acc.append(model.score(X_tr, y_tr))
    val_acc.append(model.score(X_val, y_val))

for d, tr, va in zip(depths, train_acc, val_acc):
    print(f"max_depth={d:2d}  train={tr:.3f}  val={va:.3f}")

if train_acc:
    plt.figure(figsize=(7, 4))
    plt.plot(list(depths), train_acc, "o-", label="train")
    plt.plot(list(depths), val_acc, "s-", label="validation")
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Decision Tree: model complexity vs accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
```

### ผลลัพธ์ที่คาดหวัง

```output
max_depth= 1  train=0.757  val=0.753
max_depth= 2  train=0.770  val=0.773
max_depth= 3  train=0.789  val=0.786
max_depth= 4  train=0.824  val=0.779
max_depth= 5  train=0.843  val=0.734
max_depth= 6  train=0.893  val=0.766
max_depth= 7  train=0.933  val=0.727
max_depth= 8  train=0.959  val=0.766
max_depth= 9  train=0.972  val=0.786
max_depth=10  train=0.989  val=0.753
max_depth=11  train=0.996  val=0.747
max_depth=12  train=1.000  val=0.760
max_depth=13  train=1.000  val=0.760
max_depth=14  train=1.000  val=0.760
max_depth=15  train=1.000  val=0.760
```

พร้อมกราฟเส้น 2 เส้น: เส้น train เพิ่มขึ้นจนถึง 1.0 ส่วนเส้น validation แกว่งอยู่ราว 0.73–0.79

### คำถามท้ายข้อ

1. ช่วง `max_depth` ใดที่เป็น underfitting และช่วงใดที่เป็น overfitting อธิบายจากกราฟ
2. ทำไม train accuracy ถึง 1.000 ได้ แต่ validation ไม่ดีขึ้นเลย
3. เส้น validation ขึ้นลงไม่สม่ำเสมอ (เช่น depth 8 → 9 → 10) เพราะอะไร การเลือก `max_depth` จาก validation set ชุดเดียวมีความเสี่ยงอย่างไร

## ข้อที่ 7: เลือก Hyperparameter ด้วย Cross-Validation แล้วทดสอบครั้งสุดท้าย

### โจทย์

1. ใช้ `cross_val_score` แบบ 5-fold บน **training set ทั้งหมด** (`X_train`, `y_train`) หาค่าเฉลี่ย accuracy ของ `max_depth` แต่ละค่า
2. เลือก `best_depth` ที่ได้ CV mean สูงสุด
3. ฝึกโมเดลสุดท้ายด้วย `best_depth` บน training set ทั้งหมด แล้ววัดบน test set **ครั้งเดียว**

### โค้ดเริ่มต้น

```python
cv_means = {}
for d in [1, 2, 3, 4, 5, 6, 8, None]:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = None   # TODO: cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    if scores is None:
        continue
    cv_means[d] = scores.mean()
    print(f"max_depth={str(d):4s}  CV mean={scores.mean():.3f} ± {scores.std():.3f}")

best_depth = max(cv_means, key=cv_means.get) if cv_means else 3
print("best max_depth =", best_depth)

final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42).fit(X_train, y_train)
print(f"Final test accuracy = {final_model.score(X_test, y_test):.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
max_depth=1     CV mean=0.748 ± 0.033
max_depth=2     CV mean=0.753 ± 0.031
max_depth=3     CV mean=0.740 ± 0.029
max_depth=4     CV mean=0.726 ± 0.030
max_depth=5     CV mean=0.713 ± 0.029
max_depth=6     CV mean=0.704 ± 0.043
max_depth=8     CV mean=0.689 ± 0.045
max_depth=None  CV mean=0.674 ± 0.038
best max_depth = 2
Final test accuracy = 0.695
```

### คำถามท้ายข้อ

1. ค่า ± หลัง CV mean หมายถึงอะไร ทำไม cross-validation จึงน่าเชื่อถือกว่า validation set ชุดเดียวในข้อ 6
2. ถ้านักศึกษาลองทุก `max_depth` กับ **test set** โดยตรง จะพบว่า `max_depth=4` ได้ test accuracy ประมาณ 0.80 ซึ่งสูงกว่า 0.695 — ทำไมเราจึง **ไม่ควร** รายงานตัวเลข 0.80 นั้น
3. สรุปตารางเปรียบเทียบ test accuracy ของ (ก) baseline (ข) rule-based ของนักศึกษา (ค) Decision Tree ที่เลือกด้วย CV พร้อมอภิปรายว่าในงานคัดกรองเบาหวานจริง นักศึกษาจะเลือกแนวทางใด หรือผสมทั้งสองอย่างไร

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | คะแนน | สิ่งที่ตรวจในใบงานนี้ |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | 2 | คำถามท้ายข้อ 1–2: ประเภทการเรียนรู้, forward/backward chaining, Modus Ponens |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | 2 | ตารางจับคู่ 10 ปัญหา (ข้อ 1) และเหตุผลการเลือกระบบกฎ vs ML (ข้อ 5, 7) |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | 2 | แบ่ง train/validation/test ถูกต้อง ไม่ใช้ test set ในการเลือกโมเดล (ข้อ 3, 6, 7) |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | 3 | กฎใน knowledge base, `rule_based_predict`, Decision Tree, กราฟ overfitting, `cross_val_score` รันได้และผลถูกต้อง |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | 1 | ตารางเปรียบเทียบและอภิปรายในข้อ 7 |
| **รวม** | **10** | |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab01_รหัสนักศึกษา.ipynb` ที่รันครบทุกเซลล์ตั้งแต่ข้อ 0 ถึงข้อ 7 (Runtime → Run all) โดยไม่มี error
2. คำตอบคำถามท้ายข้อทุกข้อ เขียนในเซลล์ Markdown ใต้โค้ดของข้อนั้น
3. กราฟ overfitting จากข้อ 6 และตารางเปรียบเทียบ test accuracy จากข้อ 7
4. ส่งผ่านระบบ e-learning ของรายวิชาภายในกำหนด (ก่อนเริ่มคาบสัปดาห์ที่ 3)
