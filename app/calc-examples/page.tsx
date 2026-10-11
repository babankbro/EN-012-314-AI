import Link from "next/link";
import { Calculator } from "lucide-react";
import { getCalcExamplesContent } from "@/lib/content";
import { course } from "@/lib/course";
import MarkdownContent from "@/components/MarkdownContent";

export const metadata = {
  title: `ตัวอย่างโจทย์คำนวณ | ${course.code}`,
  description: "โจทย์คำนวณแนวข้อสอบกลางภาค ข้อ 6–14 พร้อม infographic วิธีทำทีละขั้น และสรุปคำตอบ",
};

export default async function CalcExamplesPage() {
  const markdown = await getCalcExamplesContent();

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6">
        <Link href="/" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
          หน้าหลัก
        </Link>
        <span>/</span>
        <span className="text-gray-900 dark:text-white font-medium">ตัวอย่างโจทย์คำนวณ</span>
      </nav>

      <div className="mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-50 dark:bg-sky-900/30 text-sky-700 dark:text-sky-300 text-xs font-medium mb-3">
          <Calculator className="w-3.5 h-3.5" /> แนวข้อสอบกลางภาค
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-1">ตัวอย่างโจทย์คำนวณพร้อมวิธีทำ</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {course.code} {course.titleTh} — ข้อ 6–14 แบบ (ก) และ (ข) พร้อม infographic
        </p>
      </div>

      <div className="h-px bg-gray-200 dark:bg-gray-800 mb-8" />

      <article className="max-w-none">
        <MarkdownContent content={markdown} />
      </article>
    </div>
  );
}
