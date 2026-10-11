"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  ChevronDown,
  ChevronRight,
  FileText,
  FlaskConical,
  ClipboardCheck,
  ExternalLink,
  PenSquare,
} from "lucide-react";
import { modules } from "@/lib/modules-data";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

function SidebarSection({
  title,
  icon,
  children,
  defaultOpen = false,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="mb-1">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 w-full px-3 py-2 text-sm font-semibold text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
      >
        {icon}
        <span className="flex-1 text-left">{title}</span>
        {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
      {isOpen && <div className="ml-2 mt-0.5">{children}</div>}
    </div>
  );
}

function SidebarLink({
  href,
  label,
  isActive,
  external,
  onClick,
}: {
  href: string;
  label: string;
  isActive?: boolean;
  external?: boolean;
  onClick?: () => void;
}) {
  const baseClass = "flex items-center gap-2 px-3 py-1.5 text-[13px] rounded-lg transition-colors w-full";
  const activeClass = "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium";
  const inactiveClass =
    "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800";

  if (external) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className={`${baseClass} ${inactiveClass}`} onClick={onClick}>
        <span className="truncate flex-1">{label}</span>
        <ExternalLink className="w-3 h-3 flex-shrink-0 opacity-50" />
      </a>
    );
  }

  return (
    <Link href={href} className={`${baseClass} ${isActive ? activeClass : inactiveClass}`} onClick={onClick}>
      <span className="truncate">{label}</span>
    </Link>
  );
}

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const pathname = usePathname();

  return (
    <>
      {isOpen && <div className="fixed inset-0 bg-black/40 z-40 lg:hidden" onClick={onClose} />}

      <aside
        className={`
          fixed top-14 bottom-0 left-0 z-40 w-72 bg-white dark:bg-gray-900
          border-r border-gray-200 dark:border-gray-800
          overflow-y-auto overscroll-contain
          transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:sticky lg:top-14 lg:h-[calc(100vh-3.5rem)]
          print:hidden
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        <nav className="p-3 pb-20">
          <SidebarLink href="/" label="ภาพรวมรายวิชา" isActive={pathname === "/"} onClick={onClose} />
          <SidebarLink href="/exam-prep" label="🎯 แนวทางเตรียมสอบ" isActive={pathname === "/exam-prep"} onClick={onClose} />

          <div className="h-px bg-gray-200 dark:bg-gray-800 my-2" />

          <SidebarSection title="บทเรียน (Modules)" icon={<BookOpen className="w-4 h-4" />} defaultOpen>
            {modules.map((mod) => (
              <SidebarLink
                key={mod.id}
                href={`/module/${mod.id}`}
                label={`${mod.id}. ${mod.titleTh}`}
                isActive={pathname === `/module/${mod.id}`}
                onClick={onClose}
              />
            ))}
          </SidebarSection>

          <SidebarSection title="ใบงาน (Labs)" icon={<FlaskConical className="w-4 h-4" />}>
            {modules.map((mod) => (
              <SidebarLink
                key={mod.id}
                href={`/lab/${mod.id}`}
                label={`Lab ${mod.id}: ${mod.labTitle}`}
                isActive={pathname === `/lab/${mod.id}`}
                onClick={onClose}
              />
            ))}
          </SidebarSection>

          <SidebarSection title="แบบทบทวน (Review Sheets)" icon={<ClipboardCheck className="w-4 h-4" />}>
            {modules.map((mod) => (
              <SidebarLink
                key={mod.id}
                href={`/api/content/SHEET/${mod.reviewSheet}`}
                label={`Review ${mod.id}: ${mod.titleTh}`}
                external
                onClick={onClose}
              />
            ))}
          </SidebarSection>

          <SidebarSection title="แบบทดสอบ (Exams)" icon={<PenSquare className="w-4 h-4" />}>
            {modules.map((mod) => (
              <SidebarLink
                key={mod.id}
                href={`/exam/${mod.id}`}
                label={`Exam ${mod.id}: ${mod.titleTh}`}
                isActive={pathname === `/exam/${mod.id}`}
                onClick={onClose}
              />
            ))}
          </SidebarSection>

          <div className="h-px bg-gray-200 dark:bg-gray-800 my-2" />

          <SidebarSection title="แหล่งข้อมูล" icon={<FileText className="w-4 h-4" />}>
            <SidebarLink href="/api/content/datasets/diabetes_binary_classification.csv" label="Dataset: Diabetes (ตัวอย่าง 15 แถว)" external onClick={onClose} />
            <SidebarLink href="/api/content/datasets/pima_indians_diabetes_real.csv" label="Dataset: Pima Indians Diabetes" external onClick={onClose} />
            <SidebarLink href="https://colab.research.google.com/" label="Google Colab" external onClick={onClose} />
            <SidebarLink href="https://scikit-learn.org/stable/" label="scikit-learn Documentation" external onClick={onClose} />
            <SidebarLink href="https://www.tensorflow.org/" label="TensorFlow / Keras" external onClick={onClose} />
            <SidebarLink href="https://pytorch.org/" label="PyTorch" external onClick={onClose} />
            <SidebarLink href="https://archive.ics.uci.edu/" label="UCI ML Repository" external onClick={onClose} />
            <SidebarLink href="https://www.kaggle.com/datasets" label="Kaggle Datasets" external onClick={onClose} />
          </SidebarSection>
        </nav>
      </aside>
    </>
  );
}
