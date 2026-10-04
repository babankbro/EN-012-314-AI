import Link from "next/link";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { modules } from "@/lib/modules-data";

/** ลิงก์ก่อนหน้า/ถัดไป สำหรับหน้าบทเรียน (/module) และใบงาน (/lab) */
export default function PrevNextNav({ id, kind }: { id: string; kind: "module" | "lab" }) {
  const index = modules.findIndex((m) => m.id === id);
  const prev = index > 0 ? modules[index - 1] : null;
  const next = index < modules.length - 1 ? modules[index + 1] : null;
  const label = (m: (typeof modules)[number]) =>
    kind === "module" ? `${m.id}. ${m.titleTh}` : `Lab ${m.id}: ${m.labTitle}`;
  const hover = kind === "module" ? "hover:border-blue-300 dark:hover:border-blue-700" : "hover:border-green-300 dark:hover:border-green-700";

  return (
    <div className="flex flex-col sm:flex-row items-stretch gap-4">
      {prev ? (
        <Link
          href={`/${kind}/${prev.id}`}
          className={`flex-1 group flex items-center gap-3 p-4 rounded-xl border border-gray-200 dark:border-gray-800 ${hover} hover:shadow-md transition-all`}
        >
          <ArrowLeft className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors flex-shrink-0" />
          <div className="text-right flex-1">
            <p className="text-xs text-gray-400 dark:text-gray-500">ก่อนหน้า</p>
            <p className="text-sm font-medium text-gray-900 dark:text-white">{label(prev)}</p>
          </div>
        </Link>
      ) : (
        <div className="flex-1" />
      )}
      {next ? (
        <Link
          href={`/${kind}/${next.id}`}
          className={`flex-1 group flex items-center gap-3 p-4 rounded-xl border border-gray-200 dark:border-gray-800 ${hover} hover:shadow-md transition-all`}
        >
          <div className="flex-1">
            <p className="text-xs text-gray-400 dark:text-gray-500">ถัดไป</p>
            <p className="text-sm font-medium text-gray-900 dark:text-white">{label(next)}</p>
          </div>
          <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 transition-colors flex-shrink-0" />
        </Link>
      ) : (
        <div className="flex-1" />
      )}
    </div>
  );
}
