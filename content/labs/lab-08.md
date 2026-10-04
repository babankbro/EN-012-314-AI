## ข้อมูลทั่วไปของใบงาน

| รายการ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 08: Text Classification และ Sentiment Analysis |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) |
| สัปดาห์ | 12 |
| งานตาม มคอ.3 | แบบฝึกหัด 7: NLP/NLU pipeline |
| CLO/LLO | CLO4 (LLO4-1, LLO4-2) |
| เครื่องมือ | Google Colab, Python 3, scikit-learn, pandas, PyThaiNLP, NLTK |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ + คำตอบคำถามท้ายข้อ + แบบออกแบบ pipeline ข้อที่ 5 |

### วัตถุประสงค์

1. ทำความสะอาดและตัดคำข้อความภาษาอังกฤษและภาษาไทยได้ และอธิบายผลของ stop words ต่อความหมายได้
2. คำนวณ Bag-of-Words และ TF-IDF ด้วยมือ แล้วตรวจสอบกับ scikit-learn ได้
3. สร้างและประเมินโมเดล sentiment analysis ภาษาไทย พร้อมตีความน้ำหนักคำได้
4. สร้างโมเดลจำแนกเจตนา (intent) ที่มีกลไก fallback เมื่อไม่มั่นใจได้
5. ออกแบบ NLP/NLU pipeline สำหรับสถานการณ์จริง และเปรียบเทียบกับแนวทาง LLM ได้

## ข้อที่ 0: เตรียมสภาพแวดล้อม

รันเซลล์นี้ก่อนเสมอ — ติดตั้ง PyThaiNLP (ถ้ายังไม่มี) ดาวน์โหลดพจนานุกรม WordNet และ import ไลบรารีทั้งหมด

```python
import sys
import subprocess

# ติดตั้ง PyThaiNLP ใน Colab (มีอยู่แล้วจะข้ามไป)
try:
    import pythainlp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pythainlp"], check=True)
    import pythainlp

import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

nltk.download("wordnet", quiet=True)
print("PyThaiNLP", pythainlp.__version__)
```

```output
PyThaiNLP 5.3.8
```

(เลขเวอร์ชันใน Colab อาจต่างจากนี้ได้)

## ข้อที่ 1: ทำความสะอาดและตัดคำข้อความ

### โจทย์

1. เขียนฟังก์ชัน `clean_en()` ให้ (1) เปลี่ยนเป็นตัวพิมพ์เล็ก (2) ลบ HTML tag (3) ลบ URL (4) เก็บเฉพาะตัวอักษร a–z และช่องว่าง (5) ยุบช่องว่างซ้ำ
2. ตัดคำด้วย `split()` ลบ stop words แล้วเปรียบเทียบผลของ `PorterStemmer` กับ `WordNetLemmatizer`
3. เขียนฟังก์ชัน `clean_th()` สำหรับภาษาไทย ให้ลบ URL ลบตัวเลข และยุบตัวอักษรที่ซ้ำ 3 ตัวขึ้นไปให้เหลือตัวเดียว แล้วตัดคำด้วย `word_tokenize(..., engine="newmm", keep_whitespace=False)`
4. ลบ stop words ภาษาไทยสองแบบ: แบบใช้รายการเต็ม และแบบ **เก็บคำปฏิเสธ/คำเชื่อมไว้** แล้วเปรียบเทียบ

### โค้ดเริ่มต้น

```python
en_text = "The deliveries were SUPER fast!!! Loved the 3 phones <b>so much</b> :) http://bit.ly/abc"

def clean_en(text):
    text = text.lower()
    # TODO: ลบ HTML tag ด้วย re.sub(r"<[^>]+>", " ", text)
    # TODO: ลบ URL
    # TODO: เก็บเฉพาะ a-z และช่องว่าง
    # TODO: ยุบช่องว่างซ้ำ แล้ว .strip()
    return text

cleaned = clean_en(en_text)
tokens = cleaned.split()
no_stop = tokens  # TODO: กรองคำที่อยู่ใน ENGLISH_STOP_WORDS ออก
print(cleaned)
print(tokens)
print(no_stop)

ps = PorterStemmer()
wnl = WordNetLemmatizer()
print("stem      :", [ps.stem(t) for t in no_stop])
print("lemma(v)  :", [wnl.lemmatize(t, pos="v") for t in no_stop])
print("lemma(n)  :", [wnl.lemmatize(t) for t in no_stop])
```

```python
th_text = "ร้านนี้ส่งของไวมากกก แต่สินค้าไม่ค่อยดีเลย 555 https://shop.example.com"

def clean_th(text):
    # TODO: ลบ URL
    # TODO: ลบตัวเลข re.sub(r"[0-9]+", " ", text)
    # TODO: ยุบตัวอักษรซ้ำ re.sub(r"(.)\1{2,}", r"\1", text)
    # TODO: ยุบช่องว่างซ้ำ แล้ว .strip()
    return text

th_clean = clean_th(th_text)
th_tokens = []  # TODO: word_tokenize(th_clean, engine="newmm", keep_whitespace=False)
print(th_clean)
print(th_tokens)

sw = thai_stopwords()
keep = {"ไม่", "ไม่ค่อย", "แต่"}                      # คำที่ไม่ควรลบในงาน sentiment
remove_all = [t for t in th_tokens if t not in sw]
keep_negation = th_tokens  # TODO: ลบ stop words ยกเว้นคำใน keep
print("ลบทั้งหมด     :", remove_all)
print("เก็บคำปฏิเสธ  :", keep_negation)
```

### ผลลัพธ์ที่คาดหวัง

```output
the deliveries were super fast loved the phones so much
['the', 'deliveries', 'were', 'super', 'fast', 'loved', 'the', 'phones', 'so', 'much']
['deliveries', 'super', 'fast', 'loved', 'phones']
stem      : ['deliveri', 'super', 'fast', 'love', 'phone']
lemma(v)  : ['deliveries', 'super', 'fast', 'love', 'phone']
lemma(n)  : ['delivery', 'super', 'fast', 'loved', 'phone']
```

```output
ร้านนี้ส่งของไวมาก แต่สินค้าไม่ค่อยดีเลย
['ร้าน', 'นี้', 'ส่ง', 'ของ', 'ไว', 'มาก', 'แต่', 'สินค้า', 'ไม่ค่อย', 'ดี', 'เลย']
ลบทั้งหมด     : ['ร้าน', 'ไว', 'สินค้า', 'ดี']
เก็บคำปฏิเสธ  : ['ร้าน', 'ไว', 'แต่', 'สินค้า', 'ไม่ค่อย', 'ดี']
```

(ผลการตัดคำอาจต่างเล็กน้อยตามเวอร์ชัน PyThaiNLP)

### คำถามท้ายข้อ

1. stemmer และ lemmatizer ให้ผลต่างกันอย่างไรกับคำว่า "deliveries" และ "loved" เพราะอะไร lemmatizer จึงต้องรู้ชนิดคำ (`pos`)
2. หลังลบ stop words แบบเต็ม ประโยคภาษาไทยเหลือ "ร้าน ไว สินค้า ดี" ความหมายเปลี่ยนไปอย่างไร ถ้านำไปทำ sentiment จะเกิดปัญหาอะไร
3. การลบตัวเลขทำให้ "555" (เสียงหัวเราะ) และ "3 phones" หายไป ในงาน sentiment ควรจัดการ "555" อย่างไร และอีโมติคอน `:)` ล่ะ

## ข้อที่ 2: Bag-of-Words และ TF-IDF ด้วยมือ

### โจทย์

กำหนดเอกสาร 3 ฉบับ

| เอกสาร | ข้อความ |
|---|---|
| d1 | fast delivery good product |
| d2 | slow delivery |
| d3 | good product good price |

1. เขียนตาราง Bag-of-Words (คำเรียงตามตัวอักษร) ลงในสมุดหรือเซลล์ markdown
2. คำนวณ idf ของทุกคำด้วยมือตามสูตรของ scikit-learn $\text{idf}(t) = \ln\frac{1+n}{1+\text{df}(t)} + 1$ แล้วกรอกลงใน `my_idf`
3. คำนวณเวกเตอร์ TF-IDF ของ d3 ด้วยมือ (tf × idf แล้วหารด้วย L2 norm) กรอกลงใน `my_d3`
4. รันเซลล์เพื่อตรวจกับ `CountVectorizer` และ `TfidfVectorizer`

### โค้ดเริ่มต้น

```python
docs = ["fast delivery good product", "slow delivery", "good product good price"]

# ---- ค่าที่นักศึกษาคำนวณด้วยมือ ----
my_idf = {          # TODO: แทน None ด้วยค่า idf ที่คำนวณได้ (ทศนิยม 4 ตำแหน่ง)
    "delivery": None, "fast": None, "good": None,
    "price": None, "product": None, "slow": None,
}
my_d3 = {"good": None, "price": None, "product": None}   # TODO: TF-IDF ของ d3 หลัง normalize

# ---- ตรวจด้วย scikit-learn ----
cv = CountVectorizer()
X_bow = cv.fit_transform(docs)
print(pd.DataFrame(X_bow.toarray(), columns=cv.get_feature_names_out(), index=["d1", "d2", "d3"]))

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out(), index=["d1", "d2", "d3"])
print(df_tfidf.round(4))

for word, idx in sorted(tfidf.vocabulary_.items()):
    sk = tfidf.idf_[idx]
    mine = my_idf[word]
    status = "ยังไม่กรอก" if mine is None else ("ถูก" if np.isclose(mine, sk, atol=1e-3) else "ผิด")
    print(f"idf({word}) = {sk:.4f}  ของฉัน = {mine}  -> {status}")

for word, mine in my_d3.items():
    sk = df_tfidf.loc["d3", word]
    status = "ยังไม่กรอก" if mine is None else ("ถูก" if np.isclose(mine, sk, atol=1e-3) else "ผิด")
    print(f"tfidf(d3, {word}) = {sk:.4f}  ของฉัน = {mine}  -> {status}")
```

### ผลลัพธ์ที่คาดหวัง

```output
    delivery  fast  good  price  product  slow
d1         1     1     1      0        1     0
d2         1     0     0      0        0     1
d3         0     0     2      1        1     0
    delivery    fast    good   price  product   slow
d1    0.4599  0.6047  0.4599  0.0000   0.4599  0.000
d2    0.6053  0.0000  0.0000  0.0000   0.0000  0.796
d3    0.0000  0.0000  0.7710  0.5069   0.3855  0.000
idf(delivery) = 1.2877  ของฉัน = 1.2877  -> ถูก
idf(fast) = 1.6931  ของฉัน = 1.6931  -> ถูก
idf(good) = 1.2877  ของฉัน = 1.2877  -> ถูก
idf(price) = 1.6931  ของฉัน = 1.6931  -> ถูก
idf(product) = 1.2877  ของฉัน = 1.2877  -> ถูก
idf(slow) = 1.6931  ของฉัน = 1.6931  -> ถูก
tfidf(d3, good) = 0.7710  ของฉัน = 0.771  -> ถูก
tfidf(d3, price) = 0.5069  ของฉัน = 0.5069  -> ถูก
tfidf(d3, product) = 0.3855  ของฉัน = 0.3855  -> ถูก
```

### คำถามท้ายข้อ

1. แสดงวิธีคำนวณ TF-IDF ของ d3 ทีละขั้น (tf, idf, ค่าก่อน normalize, L2 norm, ค่าหลัง normalize)
2. ใน d1 คำว่า "fast" ได้ค่าสูงกว่า "delivery" ทั้งที่ปรากฏ 1 ครั้งเท่ากัน เพราะอะไร
3. ถ้าเพิ่มเอกสาร d4 = "good good good" ค่า idf ของ "good" จะเพิ่มหรือลด อธิบายเหตุผล

## ข้อที่ 3: Sentiment Classifier รีวิวร้านอาหาร (ภาษาไทย)

### โจทย์

ใช้ชุดข้อมูลรีวิวสั่งอาหารออนไลน์ 40 ข้อความ (บวก 20 ลบ 20) ที่ให้มา

1. แบ่ง train/test 70/30 แบบ stratified (`random_state=42`)
2. สร้าง `vectorizer` เป็น `TfidfVectorizer` ที่ใช้ฟังก์ชัน `thai_tokenize` ตัดคำ (`tokenizer=thai_tokenize, token_pattern=None`) และ `ngram_range=(1, 2)`
3. สร้าง `model` เป็น `LogisticRegression(C=10, max_iter=1000)` แล้วรวมเป็น `Pipeline`
4. รายงาน `classification_report` และ confusion matrix
5. แสดงคำ/bigram ที่มีน้ำหนักดึงไปทาง neg และ pos มากที่สุด 8 อันดับ
6. เปรียบเทียบ 3 แบบด้วย 5-fold cross-validation: (ก) word + MultinomialNB (ข) word + LogisticRegression (ค) `char_wb` (2–4) + LogisticRegression

### โค้ดเริ่มต้น

```python
reviews = [
    ("อาหารอร่อยมาก ส่งไว", "pos"), ("ร้านนี้บริการดีมาก ประทับใจ", "pos"),
    ("รสชาติดี ราคาไม่แพง", "pos"), ("ของสด อร่อย จะสั่งอีกแน่นอน", "pos"),
    ("ไรเดอร์สุภาพ อาหารยังร้อนอยู่เลย", "pos"), ("ปริมาณเยอะ คุ้มค่ามาก", "pos"),
    ("กาแฟหอม รสชาติกลมกล่อม ชอบมาก", "pos"), ("แพ็คมาดี ไม่หก อร่อยด้วย", "pos"),
    ("เมนูใหม่อร่อยมาก แนะนำเลย", "pos"), ("ส่งเร็วกว่าที่คิด ประทับใจมาก", "pos"),
    ("ขนมอร่อย หวานกำลังดี", "pos"), ("พนักงานน่ารัก บริการดี", "pos"),
    ("อร่อยทุกเมนู คุ้มราคา", "pos"), ("ชอบมาก สั่งซ้ำหลายรอบแล้ว", "pos"),
    ("เส้นเหนียวนุ่ม น้ำซุปอร่อย", "pos"), ("ราคาดี ของเยอะ ประทับใจ", "pos"),
    ("ส่งไว อาหารร้อน อร่อยมาก", "pos"), ("ร้านสะอาด อาหารดี แนะนำ", "pos"),
    ("ชาไข่มุกอร่อยมาก ไข่มุกนุ่ม", "pos"), ("คุ้มค่า อร่อย จะกลับมาอีก", "pos"),
    ("อาหารเย็นชืด ไม่อร่อย", "neg"), ("รอนานมาก กว่าจะได้ของ", "neg"),
    ("ส่งของผิด ได้ไม่ครบ", "neg"), ("รสชาติแย่ เค็มเกินไป", "neg"),
    ("ราคาแพงแต่ปริมาณน้อยมาก", "neg"), ("น้ำซุปหกเลอะทั้งถุง แย่มาก", "neg"),
    ("ไรเดอร์พูดจาไม่ดี ผิดหวัง", "neg"), ("อาหารบูด กินไม่ได้", "neg"),
    ("ไม่อร่อยเลย ไม่สั่งอีกแล้ว", "neg"), ("ช้ามาก อาหารเย็นหมด", "neg"),
    ("เจอเส้นผมในอาหาร แย่มาก", "neg"), ("หวานเกินไป กินไม่ลง", "neg"),
    ("บริการแย่ ร้านไม่รับผิดชอบ", "neg"), ("ไม่คุ้มราคา ผิดหวังมาก", "neg"),
    ("กาแฟจืด ไม่หอมเลย", "neg"), ("ของไม่สด กลิ่นแปลก", "neg"),
    ("ยกเลิกออเดอร์โดยไม่แจ้ง แย่มาก", "neg"), ("รอเกินชั่วโมง ผิดหวัง", "neg"),
    ("ได้ของไม่ครบ ติดต่อร้านไม่ได้", "neg"), ("เส้นอืด น้ำซุปจืด ไม่อร่อย", "neg"),
]
df = pd.DataFrame(reviews, columns=["text", "label"])
print(df["label"].value_counts())

def thai_tokenize(text):
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]

print(thai_tokenize("ไม่อร่อยเลย ไม่สั่งอีกแล้ว"))

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, stratify=df["label"], random_state=42)

vectorizer = None  # TODO: TfidfVectorizer(tokenizer=thai_tokenize, token_pattern=None, ngram_range=(1, 2))
model = None       # TODO: LogisticRegression(C=10, max_iter=1000)

clf = None
if vectorizer is not None and model is not None:
    clf = Pipeline([("vec", vectorizer), ("model", model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=["neg", "pos"]))
```

```python
# คำที่มีน้ำหนักมากที่สุด (coef_ < 0 ดึงไปทาง classes_[0] = 'neg', > 0 ดึงไปทาง 'pos')
if clf is not None:
    vec = clf.named_steps["vec"]
    lr = clf.named_steps["model"]
    coef = pd.Series(lr.coef_[0], index=vec.get_feature_names_out()).sort_values()
    print("ดึงไปทาง neg:\n", coef.head(8).round(3).to_string())
    print("ดึงไปทาง pos:\n", coef.tail(8).round(3).to_string())

    new = ["อร่อยมาก ส่งไวด้วย", "รอนานแถมอาหารเย็น", "ไม่อร่อย แต่ส่งไว", "อร่อยตายแหละ รอตั้งสองชั่วโมง"]
    for t, p, pr in zip(new, clf.predict(new), clf.predict_proba(new)):
        print(t, "->", p, pr.round(2))
```

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidates = {
    "word+NB": Pipeline([("vec", TfidfVectorizer(tokenizer=thai_tokenize, token_pattern=None)),
                         ("model", MultinomialNB())]),
    # TODO: เพิ่ม "word+LR" (ngram_range=(1, 2)) และ "char_wb+LR" (analyzer="char_wb", ngram_range=(2, 4))
}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, df["text"], df["label"], cv=cv)
    print(f"{name:<12} CV accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
label
pos    20
neg    20
Name: count, dtype: int64
['ไม่', 'อร่อย', 'เลย', 'ไม่', 'สั่ง', 'อีกแล้ว']
              precision    recall  f1-score   support

         neg       1.00      0.83      0.91         6
         pos       0.86      1.00      0.92         6

    accuracy                           0.92        12
   macro avg       0.93      0.92      0.92        12
weighted avg       0.93      0.92      0.92        12

[[5 1]
 [0 6]]
```

```output
ดึงไปทาง neg:
 ไม่         -2.028
ไม่ อร่อย   -1.428
เกินไป      -1.289
ได้         -1.191
ผิดหวัง     -1.173
รอ          -0.948
แย่มาก      -0.853
ของ         -0.838
ดึงไปทาง pos:
 อร่อย จะ    0.822
อีก         0.822
ร้อน        0.869
ประทับใจ    0.870
คุ้มค่า     0.935
เยอะ        0.975
อร่อย       1.134
ดี          1.339
อร่อยมาก ส่งไวด้วย -> pos [0.2 0.8]
รอนานแถมอาหารเย็น -> neg [0.78 0.22]
ไม่อร่อย แต่ส่งไว -> neg [0.52 0.48]
อร่อยตายแหละ รอตั้งสองชั่วโมง -> neg [0.62 0.38]
```

```output
word+NB      CV accuracy = 0.750 ± 0.112
word+LR      CV accuracy = 0.800 ± 0.150
char_wb+LR   CV accuracy = 0.825 ± 0.061
```

### คำถามท้ายข้อ

1. คำว่า "ไม่" มีน้ำหนักติดลบมากที่สุด สิ่งนี้บอกอะไรเกี่ยวกับการใช้ `thai_stopwords()` ในงานนี้ (เชื่อมโยงกับข้อที่ 1)
2. มีฟีเจอร์ใดบ้างที่ดูไม่สมเหตุสมผล (เช่น "ได้", "ของ") ทำไมโมเดลจึงให้น้ำหนักคำเหล่านี้ และจะแก้ไขอย่างไร
3. ประโยค "อร่อยตายแหละ รอตั้งสองชั่วโมง" เป็นการประชด โมเดลตอบถูกหรือไม่ และถูกเพราะ "เข้าใจการประชด" จริงหรือเพราะเหตุใด
4. จากผล cross-validation แบบไหนดีที่สุด ค่า ± บอกอะไร และข้อมูล 40 ข้อความเพียงพอที่จะสรุปหรือไม่

## ข้อที่ 4: Intent Classifier พร้อม Fallback

### โจทย์

สร้าง NLU ของ chatbot งานทะเบียนนักศึกษา ที่มี 4 intent: ทักทาย, ตารางเรียน, ลงทะเบียน, ค่าเทอม

1. สร้าง `intent_clf` เป็น `Pipeline` ของ `TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))` และ `LogisticRegression(C=10, max_iter=1000)` แล้ว fit
2. เขียนฟังก์ชัน `chatbot_intent(text, threshold)` ที่คืน `"ไม่เข้าใจ"` เมื่อความน่าจะเป็นสูงสุดต่ำกว่า threshold
3. ทดลอง threshold = 0.3, 0.5, 0.7 กับคำถามในขอบเขต (in-domain) และนอกขอบเขต (out-of-domain) แล้วสรุปอัตรา fallback

### โค้ดเริ่มต้น

```python
intent_data = [
    ("สวัสดีครับ", "ทักทาย"), ("หวัดดีค่ะ", "ทักทาย"), ("สวัสดีตอนเช้าครับ", "ทักทาย"),
    ("ดีครับแอดมิน", "ทักทาย"), ("ขอบคุณครับ", "ทักทาย"), ("ขอบคุณมากค่ะ", "ทักทาย"),
    ("ตารางเรียนเทอมนี้ดูได้ที่ไหน", "ตารางเรียน"), ("วิชา AI เรียนห้องไหน", "ตารางเรียน"),
    ("พรุ่งนี้มีเรียนกี่โมง", "ตารางเรียน"), ("ขอตารางเรียนปี 3 หน่อย", "ตารางเรียน"),
    ("วันศุกร์มีเรียนวิชาอะไรบ้าง", "ตารางเรียน"), ("ตารางสอบออกหรือยัง", "ตารางเรียน"),
    ("ลงทะเบียนเรียนวันไหน", "ลงทะเบียน"), ("เพิ่มถอนรายวิชาได้ถึงเมื่อไร", "ลงทะเบียน"),
    ("ลงทะเบียนไม่ทันทำยังไง", "ลงทะเบียน"), ("ขอเปิดรายวิชาเพิ่มได้ไหม", "ลงทะเบียน"),
    ("ระบบลงทะเบียนเข้าไม่ได้", "ลงทะเบียน"), ("ลงเรียนเกินหน่วยกิตได้ไหม", "ลงทะเบียน"),
    ("ค่าเทอมเท่าไหร่", "ค่าเทอม"), ("จ่ายค่าเทอมได้ถึงวันไหน", "ค่าเทอม"),
    ("ชำระค่าธรรมเนียมผ่านธนาคารได้ไหม", "ค่าเทอม"), ("ผ่อนค่าเทอมได้ไหมครับ", "ค่าเทอม"),
    ("ขอใบเสร็จค่าเทอม", "ค่าเทอม"), ("กู้ กยศ. จ่ายค่าเทอมยังไง", "ค่าเทอม"),
]
df_int = pd.DataFrame(intent_data, columns=["text", "intent"])

intent_clf = None  # TODO: Pipeline([("vec", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))), ("model", LogisticRegression(C=10, max_iter=1000))])
if intent_clf is not None:
    intent_clf.fit(df_int["text"], df_int["intent"])
    print(intent_clf.classes_)

def chatbot_intent(text, threshold=0.5):
    proba = intent_clf.predict_proba([text])[0]
    best = int(np.argmax(proba))
    # TODO: ถ้า proba[best] < threshold ให้ return "ไม่เข้าใจ", proba[best]
    return intent_clf.classes_[best], proba[best]
```

```python
in_domain = ["สวัสดีค่ะแอดมิน", "วิชาฐานข้อมูลเรียนกี่โมง", "ถอนรายวิชาได้ถึงวันไหน", "ค่าเทอมปีนี้กี่บาท"]
out_domain = ["ร้านข้าวมันไก่อร่อยไหม", "ฟุตบอลคืนนี้ใครชนะ", "ที่จอดรถเต็มหรือเปล่า"]

if intent_clf is not None:
    rows = []
    for th in [0.3, 0.5, 0.7]:
        for q in in_domain + out_domain:
            intent, conf = chatbot_intent(q, th)
            rows.append({"threshold": th, "query": q, "intent": intent, "conf": round(float(conf), 2),
                         "domain": "in" if q in in_domain else "out"})
    res = pd.DataFrame(rows)
    print(res[res["threshold"] == 0.5].to_string(index=False))
    res["fallback"] = res["intent"] == "ไม่เข้าใจ"
    print(res.groupby(["threshold", "domain"])["fallback"].mean().unstack())
```

### ผลลัพธ์ที่คาดหวัง

```output
['ค่าเทอม' 'ตารางเรียน' 'ทักทาย' 'ลงทะเบียน']
```

```output
 threshold                    query     intent  conf domain
       0.5          สวัสดีค่ะแอดมิน     ทักทาย  0.87     in
       0.5 วิชาฐานข้อมูลเรียนกี่โมง ตารางเรียน  0.72     in
       0.5   ถอนรายวิชาได้ถึงวันไหน  ลงทะเบียน  0.61     in
       0.5       ค่าเทอมปีนี้กี่บาท    ค่าเทอม  0.63     in
       0.5   ร้านข้าวมันไก่อร่อยไหม  ไม่เข้าใจ  0.33    out
       0.5       ฟุตบอลคืนนี้ใครชนะ  ไม่เข้าใจ  0.38    out
       0.5    ที่จอดรถเต็มหรือเปล่า  ไม่เข้าใจ  0.38    out
domain      in  out
threshold          
0.3        0.0  0.0
0.5        0.0  1.0
0.7        0.5  1.0
```

(ตารางภาษาไทยอาจจัดคอลัมน์ไม่ตรงกันเพราะสระบน/ล่างไม่กินความกว้าง — ไม่ใช่ข้อผิดพลาด)

### คำถามท้ายข้อ

1. threshold ใดเหมาะที่สุดสำหรับข้อมูลชุดนี้ อธิบาย trade-off ระหว่าง "fallback ของคำถามในขอบเขต" กับ "ตอบผิดเมื่อคำถามนอกขอบเขต"
2. ทำไมโมเดลจำแนกจึงไม่สามารถตอบ "ไม่รู้" ได้เองถ้าไม่มี threshold
3. ถ้าผู้ใช้พิมพ์ "ลงทะเบียนวิชา AI ได้วันไหน" ควรมี slot อะไรบ้างที่ bot ต้องดึงออกมา เสนอวิธีดึง slot นั้นอย่างง่าย

## ข้อที่ 5: ออกแบบ NLP/NLU Pipeline สำหรับ LINE Chatbot

### โจทย์

ร้านขายอุปกรณ์คอมพิวเตอร์ออนไลน์ในจังหวัดกาฬสินธุ์ต้องการ **LINE chatbot ภาษาไทย** สำหรับตอบลูกค้า ข้อความที่เข้ามามีทั้งภาษาทางการ ภาษาพูด คำลากเสียง ("ได้ป่าวค้าบบบ") สติกเกอร์ อีโมจิ และเลขพัสดุ ร้านมีแชตเก่าที่ยังไม่ได้ติด label ประมาณ 5,000 ข้อความ และงบประมาณจำกัด (ไม่มี GPU)

ให้ออกแบบ pipeline โดยกรอกตารางต่อไปนี้ในเซลล์ markdown (ไม่ต้องเขียนโค้ด) พร้อมวาดแผนภาพ (มือหรือ mermaid ก็ได้)

| ขั้นตอน | สิ่งที่จะทำ / เครื่องมือที่เลือก | เหตุผล |
|---|---|---|
| 1. รวบรวมข้อมูลและติด label | | |
| 2. ทำความสะอาด (อะไรลบ อะไรเก็บ) | | |
| 3. ตัดคำ / หน่วยของฟีเจอร์ | | |
| 4. Stop words / normalization | | |
| 5. Feature extraction | | |
| 6. Intent model + fallback | | |
| 7. Slot / entity ที่ต้องดึง | | |
| 8. การประเมินผลและการปรับปรุงต่อเนื่อง | | |

### คำถามท้ายข้อ

1. ระบุ intent อย่างน้อย 5 intent ที่ร้านนี้ควรมี พร้อมตัวอย่างข้อความ intent ละ 2 ตัวอย่าง
2. จะจัดการกับข้อมูลส่วนบุคคลในแชตเก่า (ชื่อ เบอร์โทร ที่อยู่) อย่างไรก่อนนำไปฝึกโมเดล
3. ใช้ metric ใดประเมิน intent classifier (accuracy, macro F1, อัตรา fallback) เพราะเหตุใด

## ข้อที่ 6 (ไม่บังคับ): เปรียบเทียบกับ LLM Prompt

### โจทย์

ไม่ต้องเรียก API จริง ให้ **เขียน prompt** สำหรับ LLM ที่ทำงานเดียวกับข้อที่ 3 (sentiment รีวิวร้านอาหาร) ทั้งแบบ zero-shot และ few-shot (3 ตัวอย่าง) โดยกำหนดให้ตอบเป็น JSON `{"label": "pos|neg", "reason": "..."}` เก็บ prompt ไว้ในตัวแปรด้านล่าง แล้วอภิปรายคำถามท้ายข้อ

```python
zero_shot_prompt = """
TODO: เขียน prompt แบบ zero-shot
รีวิว: {review}
"""

few_shot_prompt = """
TODO: เขียน prompt แบบ few-shot พร้อมตัวอย่าง 3 ตัวอย่าง
รีวิว: {review}
"""

test_review = "อร่อยตายแหละ รอตั้งสองชั่วโมง"
# ใช้ .replace() แทน .format() เพราะ prompt ที่มี JSON {...} จะทำให้ .format() error
print(zero_shot_prompt.replace("{review}", test_review))
print(few_shot_prompt.replace("{review}", test_review))
```

### ผลลัพธ์ที่คาดหวัง

prompt ทั้งสองแบบที่แทนรีวิวทดสอบลงไปแล้ว (เนื้อหาขึ้นกับการออกแบบของนักศึกษา) — ถ้าต้องการทดลองจริง สามารถคัดลอก prompt ไปวางในแชตบอตที่มหาวิทยาลัยอนุญาตให้ใช้ (ห้ามใส่ข้อมูลส่วนบุคคลของลูกค้าจริง)

### คำถามท้ายข้อ

1. ในข้อความแบบไหนที่คาดว่า LLM จะดีกว่าโมเดลในข้อที่ 3 อย่างชัดเจน (ยกตัวอย่าง 2 ข้อความ)
2. ถ้าร้านในข้อที่ 5 ต้องประมวลผล 5,000 ข้อความต่อวันโดยไม่มี GPU จะใช้ LLM, pipeline ดั้งเดิม หรือผสมกันอย่างไร พิจารณาต้นทุน latency ความเป็นส่วนตัว และ explainability
3. ทำไมจึงต้องบังคับรูปแบบคำตอบเป็น JSON และควรทำอย่างไรเมื่อ LLM ตอบไม่ตรงรูปแบบ

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | คะแนน | ข้อที่เกี่ยวข้อง |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง (stop words, stemming/lemmatization, TF-IDF, fallback) | 2 | 1, 2, 4 |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม (word vs char n-gram, NB vs LR, threshold, LLM vs classic) | 2 | 3, 4, 6 |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | 2 | 5 |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง (TF-IDF ด้วยมือ, classifier, metrics) | 3 | 1, 2, 3, 4 |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | 1 | คำถามท้ายข้อทั้งหมด |
| **รวม** | **10** | |

ข้อที่ 6 เป็นงานไม่บังคับ ใช้ชดเชยคะแนนในประเด็น "วิเคราะห์ปัญหาและเลือกเทคนิค" ได้ไม่เกิน 1 คะแนน

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab08_<รหัสนักศึกษา>.ipynb` ที่รันครบทุกเซลล์ (Runtime → Run all) และเห็นผลลัพธ์ทุกเซลล์
2. คำตอบคำถามท้ายข้อทุกข้อ เขียนในเซลล์ markdown ใต้แต่ละข้อ
3. ตารางออกแบบ pipeline และแผนภาพของข้อที่ 5
4. (ไม่บังคับ) prompt และคำอภิปรายของข้อที่ 6
