import { notFound } from "next/navigation";
import Link from "next/link";
import { BookOpen, ClipboardCheck, Download, PenSquare } from "lucide-react";
import { modules, getModuleById } from "@/lib/modules-data";
import { getLabContent } from "@/lib/content";
import { course } from "@/lib/course";
import MarkdownContent from "@/components/MarkdownContent";
import PrevNextNav from "@/components/PrevNextNav";

interface PageProps {
  params: { id: string };
}

export function generateStaticParams() {
  return modules.map((mod) => ({ id: mod.id }));
}

export function generateMetadata({ params }: PageProps) {
  const mod = getModuleById(params.id);
  if (!mod) return { title: "Not Found" };
  return {
    title: `Lab ${mod.id}: ${mod.labTitle} | ${course.code}`,
    description: `ใบงานที่ ${mod.id} - ${mod.labTitle}`,
  };
}

export default async function LabPage({ params }: PageProps) {
  const mod = getModuleById(params.id);
  if (!mod) notFound();

  const markdown = await getLabContent(params.id);

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6">
        <Link href="/" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
          หน้าหลัก
        </Link>
        <span>/</span>
        <Link href={`/module/${mod.id}`} className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
          บทที่ {mod.id}
        </Link>
        <span>/</span>
        <span className="text-gray-900 dark:text-white font-medium">Lab {mod.id}</span>
      </nav>

      <div className="mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs font-medium mb-3">
          ใบงาน (Lab) · สัปดาห์ที่ {mod.week}
        </div>
        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white mb-1">
          Lab {mod.id}: {mod.labTitle}
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {mod.titleEn} · {mod.clo} ({mod.llos.join(", ")})
        </p>
      </div>

      <div className="flex flex-wrap gap-2 mb-8">
        <Link
          href={`/module/${mod.id}`}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
        >
          <BookOpen className="w-4 h-4" /> เนื้อหาบท {mod.id}
        </Link>
        <a
          href={`/api/content/notebooks/lab-${mod.id}.ipynb`}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold rounded-lg bg-green-600 dark:bg-green-700 text-white border border-green-700 dark:border-green-600 hover:bg-green-700 dark:hover:bg-green-600 transition-colors shadow-sm"
        >
          <Download className="w-4 h-4" /> ดาวน์โหลด Notebook (.ipynb)
        </a>
        <a
          href={`/api/content/SHEET/${mod.reviewSheet}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg bg-orange-50 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300 border border-orange-200 dark:border-orange-800 hover:bg-orange-100 dark:hover:bg-orange-900/50 transition-colors"
        >
          <ClipboardCheck className="w-4 h-4" /> Review Sheet
        </a>
        <Link
          href={`/exam/${mod.id}`}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800 hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
        >
          <PenSquare className="w-4 h-4" /> แบบทดสอบ
        </Link>
      </div>

      <div className="h-px bg-gray-200 dark:bg-gray-800 mb-8" />

      <article className="max-w-none">
        <MarkdownContent content={markdown} />
      </article>

      <div className="h-px bg-gray-200 dark:bg-gray-800 my-10" />

      <PrevNextNav id={mod.id} kind="lab" />
    </div>
  );
}
