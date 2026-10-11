import Link from "next/link";
import {
  BookOpen,
  FileEdit,
  ClipboardCheck,
  ArrowRight,
  GraduationCap,
  Layers,
  PenSquare,
  Target,
  Library,
  Calculator,
} from "lucide-react";
import { modules, colorMap } from "@/lib/modules-data";
import { course } from "@/lib/course";
import ModuleIcon from "@/components/ModuleIcon";

const stats = [
  { icon: BookOpen, label: `${modules.length} บทเรียน`, sub: "Modules", color: "text-blue-600 dark:text-blue-400" },
  { icon: FileEdit, label: `${modules.length} ใบงาน`, sub: "Labs + Notebooks", color: "text-green-600 dark:text-green-400" },
  { icon: ClipboardCheck, label: `${modules.length} แบบทบทวน`, sub: "Review Sheets", color: "text-orange-600 dark:text-orange-400" },
  { icon: PenSquare, label: `${modules.length} แบบทดสอบ`, sub: "Exams", color: "text-purple-600 dark:text-purple-400" },
];

const clos = [
  { id: "CLO1", text: "อธิบายหลักการพื้นฐานของ AI, Machine Learning, Deep Learning และการแทนความรู้" },
  { id: "CLO2", text: "วิเคราะห์ เตรียมข้อมูล พัฒนาแบบจำลอง ML และประเมินผลแบบจำลอง" },
  { id: "CLO3", text: "ออกแบบและประยุกต์ใช้ Neural Networks และ Deep Learning สำหรับงานภาพ" },
  { id: "CLO4", text: "ประยุกต์ใช้ AI สำหรับ Natural Language Understanding และ Game Playing ด้วย RL" },
];

const textbooks = [
  "Géron, A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly, 2022.",
  "Russell, S., & Norvig, P. Artificial Intelligence: A Modern Approach (4th ed.). Pearson, 2021.",
  "Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning. MIT Press, 2016.",
];

export default function HomePage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-10">
      {/* Hero */}
      <div className="mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-xs font-medium mb-4">
          <GraduationCap className="w-3.5 h-3.5" />
          สื่อการสอนออนไลน์ · ภาคการศึกษาที่ 1/{course.year}
        </div>
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-2">
          {course.code} {course.titleEn}
        </h1>
        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-4">
          {course.titleTh}
        </h2>
        <p className="text-gray-500 dark:text-gray-400 mb-1">จัดทำโดย</p>
        <p className="text-xl font-semibold text-gray-900 dark:text-white mb-1">{course.instructor}</p>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          {course.program} {course.faculty}
        </p>
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
          เรียนรู้ Machine Learning ตั้งแต่ภาพรวม การเตรียมข้อมูล Classification, Regression, Unsupervised Learning
          ไปจนถึง Neural Networks, CNN, NLP และ Reinforcement Learning ด้วย Python, scikit-learn และ Keras
          พร้อมใบงานที่รันบน Google Colab ได้ทันที
        </p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-12">
        {stats.map(({ icon: Icon, label, sub, color }) => (
          <div key={sub} className="flex items-center gap-3 p-3 rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
            <Icon className={`w-5 h-5 ${color}`} />
            <div>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">{label}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">{sub}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Exam prep */}
      <Link
        href="/exam-prep"
        className="flex items-center gap-4 p-5 mb-12 rounded-2xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 hover:shadow-md transition-all"
      >
        <div className="w-12 h-12 rounded-xl bg-amber-500 text-white flex items-center justify-center flex-shrink-0">
          <Target className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <p className="font-bold text-gray-900 dark:text-white">แนวทางเตรียมสอบและแนวข้อสอบ</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            หัวข้อที่ออกสอบ สูตรที่ต้องจำ วิธีทำโน้ต 6 แผ่น และแนวข้อสอบอัตนัย 20 ข้อพร้อมเฉลย
          </p>
        </div>
        <ArrowRight className="w-5 h-5 text-amber-600 dark:text-amber-400" />
      </Link>

      <Link
        href="/calc-examples"
        className="flex items-center gap-4 p-5 -mt-8 mb-12 rounded-2xl border border-sky-200 dark:border-sky-800 bg-sky-50 dark:bg-sky-900/20 hover:shadow-md transition-all"
      >
        <div className="w-12 h-12 rounded-xl bg-sky-600 text-white flex items-center justify-center flex-shrink-0">
          <Calculator className="w-6 h-6" />
        </div>
        <div className="flex-1">
          <p className="font-bold text-gray-900 dark:text-white">ตัวอย่างโจทย์คำนวณพร้อมวิธีทำ</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            โจทย์แนวข้อสอบกลางภาคข้อ 6–14 แบบ (ก) และ (ข) พร้อม infographic และสรุปคำตอบ
          </p>
        </div>
        <ArrowRight className="w-5 h-5 text-sky-600 dark:text-sky-400" />
      </Link>

      {/* Module List */}
      <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-5 flex items-center gap-2">
        <Layers className="w-5 h-5" />
        บทเรียนทั้งหมด
      </h2>

      <div className="space-y-3">
        {modules.map((mod) => {
          const colors = colorMap[mod.color] ?? colorMap.blue;
          return (
            <div
              key={mod.id}
              className={`group flex items-start gap-4 p-4 rounded-xl border ${colors.border} hover:shadow-md transition-all bg-white dark:bg-gray-900`}
            >
              <Link
                href={`/module/${mod.id}`}
                className={`flex-shrink-0 w-11 h-11 rounded-xl ${colors.bg} ${colors.text} flex flex-col items-center justify-center hover:opacity-80 transition-opacity`}
              >
                <ModuleIcon name={mod.icon} className="w-4 h-4" />
                <span className="text-[10px] font-bold leading-none mt-0.5">{mod.id}</span>
              </Link>

              <div className="flex-1 min-w-0">
                <Link href={`/module/${mod.id}`} className="block">
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <h3 className="font-semibold text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                      {mod.titleTh}
                    </h3>
                    <span className="text-[11px] text-gray-400 dark:text-gray-500">
                      สัปดาห์ที่ {mod.week} · {mod.clo}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">{mod.titleEn}</p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{mod.description}</p>
                </Link>

                <div className="flex flex-wrap gap-2 mt-3">
                  <Link
                    href={`/lab/${mod.id}`}
                    className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
                  >
                    <FileEdit className="w-3 h-3" /> Lab
                  </Link>
                  <a
                    href={`/api/content/SHEET/${mod.reviewSheet}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 hover:bg-orange-100 dark:hover:bg-orange-900/50 transition-colors"
                  >
                    <ClipboardCheck className="w-3 h-3" /> Review
                  </a>
                  <Link
                    href={`/exam/${mod.id}`}
                    className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-semibold rounded-md bg-purple-600 dark:bg-purple-700 text-white hover:bg-purple-700 dark:hover:bg-purple-600 transition-colors"
                  >
                    <PenSquare className="w-3 h-3" /> Exam
                  </Link>
                </div>
              </div>

              <Link href={`/module/${mod.id}`} className="flex-shrink-0 mt-1" aria-label={`ไปที่บทที่ ${mod.id}`}>
                <ArrowRight className="w-5 h-5 text-gray-300 dark:text-gray-600 hover:text-blue-500 dark:hover:text-blue-400 transition-colors" />
              </Link>
            </div>
          );
        })}
      </div>

      {/* CLOs */}
      <h2 className="text-xl font-bold text-gray-900 dark:text-white mt-12 mb-5 flex items-center gap-2">
        <Target className="w-5 h-5" />
        ผลลัพธ์การเรียนรู้ที่เกี่ยวข้อง (CLOs)
      </h2>
      <div className="grid md:grid-cols-2 gap-3">
        {clos.map((clo) => {
          const related = modules.filter((m) => m.clo === clo.id).map((m) => m.id);
          return (
            <div key={clo.id} className="p-4 rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
              <p className="text-sm font-bold text-violet-700 dark:text-violet-300 mb-1">{clo.id}</p>
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">{clo.text}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {related.length > 0 ? `บทที่ ${related.join(", ")}` : "บทนำ AI (01 Introduction)"}
              </p>
            </div>
          );
        })}
      </div>

      {/* Textbooks */}
      <div className="mt-12 p-6 rounded-2xl bg-gradient-to-r from-violet-600 to-blue-600 text-white">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center flex-shrink-0">
            <Library className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-bold mb-2">ตำราหลักของรายวิชา</h3>
            <ul className="space-y-1.5 text-sm text-blue-50">
              {textbooks.map((t) => (
                <li key={t}>• {t}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
