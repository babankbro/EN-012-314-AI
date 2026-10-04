import { readFile } from "fs/promises";
import path from "path";

const CONTENT_DIR = path.join(process.cwd(), "content");

export interface ExamChoice {
  key: string;
  text: string;
}

export interface ExamQuestion {
  id: number;
  question: string;
  code?: string;
  choices: ExamChoice[];
  answer: string;
  explanation: string;
}

export interface ExamData {
  moduleId: string;
  title: string;
  titleTh: string;
  timeMinutes: number;
  questions: ExamQuestion[];
}

async function readMarkdown(subdir: string, filename: string, fallback: string) {
  try {
    return await readFile(path.join(CONTENT_DIR, subdir, filename), "utf-8");
  } catch {
    return fallback;
  }
}

export function getModuleContent(id: string) {
  const num = id.padStart(2, "0");
  return readMarkdown("modules", `module-${num}.md`, `# ไม่พบเนื้อหาบทที่ ${id}`);
}

export function getLabContent(id: string) {
  const num = id.padStart(2, "0");
  return readMarkdown("labs", `lab-${num}.md`, `# ไม่พบใบงานที่ ${id}`);
}

export async function getExam(id: string): Promise<ExamData | undefined> {
  const num = id.padStart(2, "0");
  try {
    const raw = await readFile(path.join(CONTENT_DIR, "exams", `exam-${num}.json`), "utf-8");
    return JSON.parse(raw) as ExamData;
  } catch {
    return undefined;
  }
}
