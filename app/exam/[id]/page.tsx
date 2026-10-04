import { notFound } from "next/navigation";
import { modules, getModuleById } from "@/lib/modules-data";
import { getExam } from "@/lib/content";
import { course } from "@/lib/course";
import ExamClient from "@/components/ExamClient";

interface PageProps {
  params: { id: string };
}

export function generateStaticParams() {
  return modules.map((mod) => ({ id: mod.id }));
}

export function generateMetadata({ params }: PageProps) {
  const mod = getModuleById(params.id);
  if (!mod) return {};
  return {
    title: `แบบทดสอบ ${mod.id}: ${mod.titleTh} | ${course.code}`,
    description: `แบบทดสอบประจำบทที่ ${mod.id} - ${mod.titleTh}`,
  };
}

export default async function ExamPage({ params }: PageProps) {
  const mod = getModuleById(params.id);
  if (!mod) notFound();

  const exam = await getExam(params.id);
  if (!exam) notFound();

  return <ExamClient exam={exam} moduleTitleTh={mod.titleTh} />;
}
