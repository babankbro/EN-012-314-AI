import { notFound } from "next/navigation";
import Link from "next/link";
import { FileEdit, ClipboardCheck, PenSquare, CalendarDays, Target } from "lucide-react";
import { modules, getModuleById, colorMap } from "@/lib/modules-data";
import { getModuleContent } from "@/lib/content";
import { course } from "@/lib/course";
import MarkdownContent from "@/components/MarkdownContent";
import ModuleIcon from "@/components/ModuleIcon";
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
    title: `${mod.id}. ${mod.titleTh} | ${course.code}`,
    description: mod.description,
  };
}

export default async function ModulePage({ params }: PageProps) {
  const mod = getModuleById(params.id);
  if (!mod) notFound();

  const markdown = await getModuleContent(params.id);
  const colors = colorMap[mod.color] ?? colorMap.blue;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">
      <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6">
        <Link href="/" className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
          หน้าหลัก
        </Link>
        <span>/</span>
        <span className="text-gray-900 dark:text-white font-medium">บทที่ {mod.id}</span>
      </nav>

      <div className="mb-6">
        <div className="flex items-start gap-3 mb-3">
          <span className={`inline-flex items-center justify-center w-11 h-11 rounded-xl ${colors.bg} ${colors.text} flex-shrink-0`}>
            <ModuleIcon name={mod.icon} className="w-5 h-5" />
          </span>
          <div>
            <p className={`text-xs font-semibold ${colors.text}`}>บทที่ {mod.id}</p>
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-white">{mod.titleTh}</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">{mod.titleEn}</p>
          </div>
        </div>
        <p className="text-gray-600 dark:text-gray-400 sm:ml-14">{mod.description}</p>
      </div>

      {/* ความเชื่อมโยงกับ มคอ.3 */}
      <div className="sm:ml-14 mb-6 flex flex-wrap gap-2 text-xs">
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
          <CalendarDays className="w-3.5 h-3.5" /> สัปดาห์ที่ {mod.week}
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
          <Target className="w-3.5 h-3.5" /> {mod.clo} · {mod.llos.join(", ")}
        </span>
      </div>

      <div className="flex flex-wrap gap-2 mb-8 sm:ml-14">
        <Link
          href={`/lab/${mod.id}`}
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800 hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
        >
          <FileEdit className="w-4 h-4" /> ใบงาน (Lab)
        </Link>
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
          className="inline-flex items-center gap-2 px-3 py-2 text-sm font-semibold rounded-lg bg-purple-600 dark:bg-purple-700 text-white border border-purple-700 dark:border-purple-600 hover:bg-purple-700 dark:hover:bg-purple-600 transition-colors shadow-sm"
        >
          <PenSquare className="w-4 h-4" /> ทำแบบทดสอบ
        </Link>
      </div>

      <div className="h-px bg-gray-200 dark:bg-gray-800 mb-8" />

      <article className="max-w-none">
        <MarkdownContent content={markdown} />
      </article>

      <div className="h-px bg-gray-200 dark:bg-gray-800 my-10" />

      <PrevNextNav id={mod.id} kind="module" />
    </div>
  );
}
