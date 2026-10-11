# สื่อการสอนรายวิชาปัญญาประดิษฐ์ — ภาค Machine Learning

> EN-012-314 Artificial Intelligence — Next.js e-Learning Site

เว็บไซต์สื่อการสอนออนไลน์ของรายวิชา EN-012-314 ปัญญาประดิษฐ์ 10 บทเรียน (AI Hierarchy และ Machine Learning)
พร้อมใบงาน (ดาวน์โหลดเป็น Jupyter Notebook ไปเปิดใน Google Colab ได้) แบบทบทวน 1 หน้า A4 และแบบทดสอบออนไลน์พร้อมเฉลย
เนื้อหาเรียบเรียงจากคลังความรู้ `02 Machine Learning/Machine Learning Vault` และจัดลำดับตามแผนการสอนใน มคอ.3

## บทเรียน

| บท | สัปดาห์ | ชื่อบทเรียน | CLO | ใบงาน (งานตาม มคอ.3) |
|----|----|------------|-----|------|
| 00 | 1 | ลำดับชั้นของปัญญาประดิษฐ์ (AI Hierarchy) | CLO1 | แบบฝึกหัด 1 |
| 01 | 2 | ภาพรวม Machine Learning | CLO1 | แบบฝึกหัด 2 |
| 02 | 3 | ML Pipeline และการเตรียมข้อมูล | CLO2 | งานปฏิบัติ 1 |
| 03 | 4 | การจำแนกประเภทและการประเมินผล | CLO2 | แบบฝึกหัด 3 |
| 04 | 5 | การถดถอยและการประเมินโมเดล | CLO2 | งานปฏิบัติ 2 |
| 05 | 6 | Clustering, การลดมิติ และ Anomaly Detection | CLO2 | แบบฝึกหัด 4 |
| 06 | 9 | Neural Networks และ Deep Learning | CLO3 | แบบฝึกหัด 6 |
| 07 | 10 | CNN และการจำแนกภาพ | CLO3 | งานปฏิบัติ 3 |
| 08 | 12 | การประมวลผลและทำความเข้าใจภาษาธรรมชาติ | CLO4 | แบบฝึกหัด 7 |
| 09 | 13 | Reinforcement Learning และ Game Playing | CLO4 | แบบฝึกหัด 8 |

## วิธีใช้งาน

```bash
npm install
npm run dev          # เปิด http://localhost:3000
npm run notebooks    # สร้าง content/notebooks/lab-XX.ipynb จากใบงาน
npm run check        # ตรวจความครบถ้วนและรูปแบบของเนื้อหา
npm run build        # build สำหรับ production (สร้าง notebooks ให้อัตโนมัติ)
npm start
```

## โครงสร้างโปรเจกต์

```
e-learning/
├── app/
│   ├── layout.tsx              ← Root layout (ฟอนต์ Prompt, KaTeX, AppShell)
│   ├── page.tsx                ← หน้าหลัก (ภาพรวมรายวิชา + CLO)
│   ├── module/[id]/page.tsx    ← หน้าเนื้อหาบทเรียน (markdown)
│   ├── lab/[id]/page.tsx       ← หน้าใบงาน + ปุ่มดาวน์โหลด .ipynb
│   ├── exam/[id]/page.tsx      ← แบบทดสอบออนไลน์ (จับเวลา ตรวจ พิมพ์ผล)
│   ├── exam-prep/page.tsx      ← แนวทางเตรียมสอบและแนวข้อสอบอัตนัย (content/exam-prep.md)
│   ├── calc-examples/page.tsx  ← ตัวอย่างโจทย์คำนวณข้อ 6–14 พร้อม infographic (content/calc-examples.md)
│   └── api/content/[...path]/  ← ให้บริการไฟล์ใน content/ (Review Sheet, CSV, notebook)
├── components/                 ← Header, Sidebar, MarkdownContent, Mermaid, ExamClient ฯลฯ
├── lib/
│   ├── course.ts               ← ข้อมูลรายวิชา
│   ├── modules-data.ts         ← ข้อมูล 10 บท (สัปดาห์, CLO/LLO, หัวข้อ)
│   └── content.ts              ← อ่าน markdown / exam JSON
├── content/
│   ├── modules/module-XX.md    ← เนื้อหาบทเรียน
│   ├── labs/lab-XX.md          ← ใบงาน
│   ├── notebooks/lab-XX.ipynb  ← สร้างอัตโนมัติจากใบงาน
│   ├── exams/exam-XX.json      ← แบบทดสอบ 20 ข้อ/บท
│   ├── SHEET/                  ← Review Sheet HTML (+ sheet.css, _template.html)
│   └── datasets/               ← ไฟล์ CSV สำหรับใบงาน
├── public/images/mXX/          ← รูปประกอบจาก vault
├── scripts/build-notebooks.mjs ← แปลงใบงาน .md → .ipynb
├── scripts/make_calc_figures.py ← สร้าง infographic public/images/calc/*.svg (python, ต้องมีฟอนต์ Prompt)
└── CONTENT_GUIDE.md            ← คู่มือเขียน/แก้ไขเนื้อหา
```

## ฟีเจอร์

- **Docs-style Layout** — Sidebar + Content แบบเดียวกับเว็บรายวิชา EN-001-002
- **Markdown + สูตรคณิตศาสตร์ (KaTeX) + แผนภาพ Mermaid** — แก้เนื้อหาได้ที่ `content/` โดยไม่ต้องแตะโค้ด
- **ใบงาน → Notebook** — โค้ดในใบงานกลายเป็น code cell พร้อมเปิดใน Google Colab
- **แบบทดสอบออนไลน์** — จับเวลา ตรวจอัตโนมัติ แสดงเฉลยพร้อมคำอธิบาย และพิมพ์ผลได้ (เกณฑ์ผ่าน 50% ตามเกณฑ์ CLO ใน มคอ.3)
- **Review Sheet** — สรุป 1 หน้า A4 พิมพ์หรือบันทึกเป็น PDF ได้
- **ตัวอย่างโจทย์คำนวณ** — โจทย์แนวข้อสอบกลางภาคข้อ 6–14 แบบ (ก)/(ข) พร้อม infographic 15 ภาพ วิธีทำ และสรุปคำตอบ
- **แนวทางเตรียมสอบ** — สูตรที่ต้องจำ วิธีทำโน้ต 6 แผ่น และแนวข้อสอบอัตนัย 20 ข้อพร้อมเฉลย และทุกบทมีหัวข้อ "ตัวอย่างโจทย์และวิธีทำ"
- **Dark / Light Mode**, **ค้นหา (Ctrl+K)**, **Responsive**

## เทคโนโลยี

Next.js 14 (App Router) · TailwindCSS · react-markdown + remark-gfm + remark-math + rehype-katex · Mermaid · Lucide React

---

**ผู้จัดทำ:** ผศ.ดร. สรายุทธ กรวิรัตน์  
**สาขาวิชา:** วิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์และเทคโนโลยีอุตสาหกรรม มหาวิทยาลัยกาฬสินธุ์  
**ปีการศึกษา:** 2569
