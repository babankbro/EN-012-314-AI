import Link from "next/link";
import { GraduationCap } from "lucide-react";
import { getExamPrepContent } from "@/lib/content";
import { course } from "@/lib/course";
import MarkdownContent from "@/components/MarkdownContent";

export const metadata = {
  title: `แนวทางเตรียมสอบ | ${course.code}`,
  description: "คำแนะนำการเตรียมสอบ แผนอ่านหนังสือ สูตรที่ต้องจำ และแนวข้อสอบอัตนัยพร้อมเฉลย",
};

export default async function ExamPrepPage() {
  const markdown = await getExamPrepContent();

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6">
        <Link href="/" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
          หน้าหลัก
        </Link>
        <span>/</span>
        <span className="text-gray-900 dark:text-white font-medium">แนวทางเตรียมสอบ</span>
      </nav>

      <div className="mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-medium mb-3">
          <GraduationCap className="w-3.5 h-3.5" /> เตรียมสอบ
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-1">แนวทางเตรียมสอบและแนวข้อสอบ</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {course.code} {course.titleTh} — ข้อสอบอัตนัย อธิบาย คำนวณ และเขียนโค้ด
        </p>
      </div>

      <div className="h-px bg-gray-200 dark:bg-gray-800 mb-8" />

      <article className="max-w-none">
        <MarkdownContent content={markdown} />
      </article>
    </div>
  );
}
