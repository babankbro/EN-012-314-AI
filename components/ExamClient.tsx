"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Clock,
  CheckCircle2,
  XCircle,
  Printer,
  RotateCcw,
  User,
  Hash,
  Users,
  AlertTriangle,
} from "lucide-react";
import type { ExamData } from "@/lib/content";
import { course, courseLabel } from "@/lib/course";

interface ExamClientProps {
  exam: ExamData;
  moduleTitleTh: string;
}

function QuestionCode({ code }: { code?: string }) {
  if (!code) return null;
  return (
    <pre className="code-block rounded-lg mb-4 ml-9 text-[12.5px]">
      <code>{code}</code>
    </pre>
  );
}

interface StudentInfo {
  name: string;
  studentId: string;
  section: string;
}

type Phase = "info" | "quiz" | "result";

export default function ExamClient({ exam, moduleTitleTh }: ExamClientProps) {
  const [phase, setPhase] = useState<Phase>("info");
  const [studentInfo, setStudentInfo] = useState<StudentInfo>({
    name: "",
    studentId: "",
    section: "",
  });
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [timeLeft, setTimeLeft] = useState(exam.timeMinutes * 60);
  const [showConfirm, setShowConfirm] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<Date | null>(null);

  // Timer
  useEffect(() => {
    if (phase !== "quiz") return;
    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          clearInterval(timerRef.current!);
          setPhase("result");
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [phase]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleStartExam = (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentInfo.name.trim() || !studentInfo.studentId.trim()) return;
    startTimeRef.current = new Date();
    setPhase("quiz");
  };

  const handleSelectAnswer = (questionId: number, choice: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: choice }));
  };

  const handleSubmit = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    setPhase("result");
    setShowConfirm(false);
  }, []);

  const handleRetry = () => {
    setAnswers({});
    setTimeLeft(exam.timeMinutes * 60);
    setPhase("quiz");
    startTimeRef.current = new Date();
  };

  const handlePrint = () => {
    window.print();
  };

  const answeredCount = Object.keys(answers).length;
  const totalQuestions = exam.questions.length;

  // Calculate score
  const score = exam.questions.reduce((acc, q) => {
    return acc + (answers[q.id] === q.answer ? 1 : 0);
  }, 0);

  const scorePercent = Math.round((score / totalQuestions) * 100);
  const timeUsed = exam.timeMinutes * 60 - timeLeft;

  // ==================== PHASE: INFO ====================
  if (phase === "info") {
    return (
      <div className="max-w-2xl mx-auto px-6 py-8">
        <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6">
          <Link
            href="/"
            className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            หน้าหลัก
          </Link>
          <span>/</span>
          <Link
            href={`/module/${exam.moduleId}`}
            className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            บทที่ {exam.moduleId}
          </Link>
          <span>/</span>
          <span className="text-gray-900 dark:text-white">แบบทดสอบ</span>
        </nav>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl p-8 shadow-sm">
          <div className="text-center mb-8">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-medium mb-4">
              <Clock className="w-3.5 h-3.5" />
              {exam.timeMinutes} นาที | {totalQuestions} ข้อ |{" "}
              {totalQuestions} คะแนน
            </div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              แบบทดสอบประจำบทที่ {exam.moduleId}
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              {exam.title} — {exam.titleTh}
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
              วิชา {courseLabel}
            </p>
          </div>

          <form onSubmit={handleStartExam} className="space-y-5">
            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <User className="w-4 h-4" />
                ชื่อ-สกุล <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={studentInfo.name}
                onChange={(e) =>
                  setStudentInfo((prev) => ({ ...prev, name: e.target.value }))
                }
                placeholder="กรอกชื่อ-นามสกุล"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Hash className="w-4 h-4" />
                รหัสนักศึกษา <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={studentInfo.studentId}
                onChange={(e) =>
                  setStudentInfo((prev) => ({
                    ...prev,
                    studentId: e.target.value,
                  }))
                }
                placeholder="กรอกรหัสนักศึกษา"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Users className="w-4 h-4" />
                Section
              </label>
              <input
                type="text"
                value={studentInfo.section}
                onChange={(e) =>
                  setStudentInfo((prev) => ({
                    ...prev,
                    section: e.target.value,
                  }))
                }
                placeholder="เช่น 1, 2, A, B (ไม่บังคับ)"
                className="w-full px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all"
              />
            </div>

            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl p-4 text-sm text-yellow-800 dark:text-yellow-300">
              <p className="font-semibold mb-1">📋 คำชี้แจง</p>
              <p>
                จงเลือกคำตอบที่ถูกต้องที่สุดเพียงข้อเดียว (ข้อละ 1 คะแนน)
                เวลาจะเริ่มนับเมื่อกดปุ่มเริ่มทำแบบทดสอบ
              </p>
            </div>

            <button
              type="submit"
              className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-xl transition-colors shadow-sm text-lg"
            >
              เริ่มทำแบบทดสอบ
            </button>
          </form>
        </div>
      </div>
    );
  }

  // ==================== PHASE: QUIZ ====================
  if (phase === "quiz") {
    const isTimeLow = timeLeft < 300; // < 5 minutes

    return (
      <div className="max-w-3xl mx-auto px-6 py-6">
        {/* Sticky Header */}
        <div className="sticky top-14 z-30 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 -mx-6 px-6 py-3 mb-6 print:hidden">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-semibold text-gray-900 dark:text-white">
                บทที่ {exam.moduleId}
              </span>{" "}
              — {moduleTitleTh}
            </div>
            <div
              className={`flex items-center gap-1.5 text-sm font-mono font-bold px-3 py-1 rounded-lg ${
                isTimeLow
                  ? "bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 animate-pulse"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"
              }`}
            >
              <Clock className="w-4 h-4" />
              {formatTime(timeLeft)}
            </div>
          </div>

          {/* Progress bar */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 transition-all duration-300 rounded-full"
                style={{
                  width: `${(answeredCount / totalQuestions) * 100}%`,
                }}
              />
            </div>
            <span className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
              {answeredCount}/{totalQuestions} ข้อ
            </span>
          </div>
        </div>

        {/* Questions */}
        <div className="space-y-6">
          {exam.questions.map((q, idx) => (
            <div
              key={q.id}
              id={`q-${q.id}`}
              className={`bg-white dark:bg-gray-900 border rounded-xl p-5 transition-all ${
                answers[q.id]
                  ? "border-purple-200 dark:border-purple-800"
                  : "border-gray-200 dark:border-gray-800"
              }`}
            >
              <p className="font-semibold text-gray-900 dark:text-white mb-4">
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs font-bold mr-2">
                  {q.id}
                </span>
                {q.question}
              </p>
              <QuestionCode code={q.code} />

              <div className="space-y-2 ml-9">
                {q.choices.map((c) => {
                  const isSelected = answers[q.id] === c.key;
                  return (
                    <label
                      key={c.key}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-all ${
                        isSelected
                          ? "border-purple-400 dark:border-purple-600 bg-purple-50 dark:bg-purple-900/20"
                          : "border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800"
                      }`}
                    >
                      <input
                        type="radio"
                        name={`q-${q.id}`}
                        value={c.key}
                        checked={isSelected}
                        onChange={() => handleSelectAnswer(q.id, c.key)}
                        className="mt-0.5 accent-purple-600"
                      />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        <span className="font-semibold mr-1">{c.key}.</span>
                        {c.text}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Submit area */}
        <div className="mt-8 print:hidden">
          {answeredCount < totalQuestions && (
            <p className="text-sm text-yellow-600 dark:text-yellow-400 flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4" />
              ยังตอบไม่ครบ ({totalQuestions - answeredCount} ข้อที่ยังไม่ได้ตอบ)
            </p>
          )}

          {showConfirm ? (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-800 rounded-xl p-5">
              <p className="font-semibold text-yellow-800 dark:text-yellow-300 mb-3">
                ยืนยันส่งคำตอบ?
              </p>
              <p className="text-sm text-yellow-700 dark:text-yellow-400 mb-4">
                ตอบแล้ว {answeredCount}/{totalQuestions} ข้อ —
                เมื่อส่งแล้วจะไม่สามารถแก้ไขได้
              </p>
              <div className="flex gap-3">
                <button
                  onClick={handleSubmit}
                  className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
                >
                  ยืนยันส่ง
                </button>
                <button
                  onClick={() => setShowConfirm(false)}
                  className="px-6 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-semibold rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                >
                  ย้อนกลับ
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setShowConfirm(true)}
              className="w-full py-3 px-6 bg-purple-600 hover:bg-purple-700 text-white font-semibold rounded-xl transition-colors shadow-sm text-lg"
            >
              ส่งคำตอบ
            </button>
          )}
        </div>

        {/* Question navigator (mini) */}
        <div className="sticky bottom-0 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm border-t border-gray-200 dark:border-gray-800 -mx-6 px-6 py-3 mt-6 print:hidden">
          <div className="flex flex-wrap gap-1.5 justify-center">
            {exam.questions.map((q) => (
              <a
                key={q.id}
                href={`#q-${q.id}`}
                className={`w-8 h-8 flex items-center justify-center text-xs font-bold rounded-lg transition-colors ${
                  answers[q.id]
                    ? "bg-purple-500 text-white"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700"
                }`}
              >
                {q.id}
              </a>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // ==================== PHASE: RESULT ====================
  const gradeColor =
    scorePercent >= 80
      ? "text-green-600 dark:text-green-400"
      : scorePercent >= 50
      ? "text-yellow-600 dark:text-yellow-400"
      : "text-red-600 dark:text-red-400";

  const gradeLabel =
    scorePercent >= 80
      ? "ดีมาก"
      : scorePercent >= 50
      ? "ผ่าน"
      : "ไม่ผ่าน";

  const gradeBg =
    scorePercent >= 80
      ? "bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800"
      : scorePercent >= 50
      ? "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800"
      : "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800";

  const now = new Date();
  const dateStr = now.toLocaleDateString("th-TH", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const timeStr = now.toLocaleTimeString("th-TH", {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="max-w-3xl mx-auto px-6 py-8" id="exam-result">
      {/* Print header (hidden on screen, visible when printing) */}
      <div className="hidden print:block mb-6 text-center border-b-2 border-gray-900 pb-4">
        <h1 className="text-xl font-bold">ผลการทดสอบ</h1>
        <p className="text-sm">
          วิชา {courseLabel}
        </p>
        <p className="text-sm">
          {course.instructor} | {course.program} | ปีการศึกษา {course.year}
        </p>
      </div>

      {/* Screen breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 mb-6 print:hidden">
        <Link
          href="/"
          className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
        >
          หน้าหลัก
        </Link>
        <span>/</span>
        <Link
          href={`/module/${exam.moduleId}`}
          className="hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
        >
          บทที่ {exam.moduleId}
        </Link>
        <span>/</span>
        <span className="text-gray-900 dark:text-white">ผลการทดสอบ</span>
      </nav>

      {/* Score summary card */}
      <div
        className={`border rounded-2xl p-6 mb-8 ${gradeBg}`}
      >
        <div className="flex flex-col md:flex-row items-center gap-6">
          {/* Score circle */}
          <div className="relative w-32 h-32 flex-shrink-0">
            <svg className="w-full h-full" viewBox="0 0 120 120">
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="currentColor"
                strokeWidth="8"
                className="text-gray-200 dark:text-gray-700"
              />
              <circle
                cx="60"
                cy="60"
                r="52"
                fill="none"
                stroke="currentColor"
                strokeWidth="8"
                strokeDasharray={`${(scorePercent / 100) * 327} 327`}
                strokeLinecap="round"
                transform="rotate(-90 60 60)"
                className={gradeColor}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className={`text-3xl font-bold ${gradeColor}`}>
                {score}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                / {totalQuestions}
              </span>
            </div>
          </div>

          {/* Info */}
          <div className="flex-1 text-center md:text-left">
            <h1 className="text-xl font-bold text-gray-900 dark:text-white mb-1">
              ผลการทดสอบบทที่ {exam.moduleId}: {moduleTitleTh}
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              {exam.title}
            </p>

            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
              <div>
                <span className="text-gray-500 dark:text-gray-400">ชื่อ-สกุล:</span>{" "}
                <span className="font-medium text-gray-900 dark:text-white">
                  {studentInfo.name}
                </span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">รหัส:</span>{" "}
                <span className="font-medium text-gray-900 dark:text-white">
                  {studentInfo.studentId}
                </span>
              </div>
              {studentInfo.section && (
                <div>
                  <span className="text-gray-500 dark:text-gray-400">
                    Section:
                  </span>{" "}
                  <span className="font-medium text-gray-900 dark:text-white">
                    {studentInfo.section}
                  </span>
                </div>
              )}
              <div>
                <span className="text-gray-500 dark:text-gray-400">
                  คะแนน:
                </span>{" "}
                <span className={`font-bold ${gradeColor}`}>
                  {score}/{totalQuestions} ({scorePercent}%)
                </span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">ผลสอบ:</span>{" "}
                <span className={`font-bold ${gradeColor}`}>{gradeLabel}</span>
              </div>
              <div>
                <span className="text-gray-500 dark:text-gray-400">
                  เวลาที่ใช้:
                </span>{" "}
                <span className="font-medium text-gray-900 dark:text-white">
                  {formatTime(timeUsed)}
                </span>
              </div>
              <div className="col-span-2">
                <span className="text-gray-500 dark:text-gray-400">
                  วันที่สอบ:
                </span>{" "}
                <span className="font-medium text-gray-900 dark:text-white">
                  {dateStr} เวลา {timeStr}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-3 mb-8 print:hidden">
        <button
          onClick={handlePrint}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors shadow-sm"
        >
          <Printer className="w-4 h-4" /> พิมพ์ผลการทดสอบ
        </button>
        <button
          onClick={handleRetry}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-semibold rounded-xl hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
        >
          <RotateCcw className="w-4 h-4" /> ทำใหม่อีกครั้ง
        </button>
        <Link
          href={`/module/${exam.moduleId}`}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 font-semibold rounded-xl hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> กลับไปบทเรียน
        </Link>
      </div>

      {/* Detailed answers */}
      <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
        เฉลยและคำอธิบาย
      </h2>

      <div className="space-y-4">
        {exam.questions.map((q) => {
          const userAnswer = answers[q.id] || "";
          const isCorrect = userAnswer === q.answer;
          const didAnswer = userAnswer !== "";

          return (
            <div
              key={q.id}
              className={`border rounded-xl p-4 ${
                isCorrect
                  ? "border-green-200 dark:border-green-800 bg-green-50/50 dark:bg-green-900/10"
                  : "border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-900/10"
              }`}
            >
              <div className="flex items-start gap-3 mb-3">
                {isCorrect ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5 print:text-green-700" />
                ) : (
                  <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5 print:text-red-700" />
                )}
                <p className="font-semibold text-gray-900 dark:text-white text-sm">
                  <span className="text-gray-500 dark:text-gray-400 mr-1">
                    ข้อ {q.id}.
                  </span>
                  {q.question}
                </p>
              </div>
              <QuestionCode code={q.code} />

              <div className="ml-8 space-y-1.5">
                {q.choices.map((c) => {
                  const isAnswer = c.key === q.answer;
                  const isUserPick = c.key === userAnswer;
                  let choiceClass =
                    "text-sm px-3 py-1.5 rounded-lg border ";

                  if (isAnswer) {
                    choiceClass +=
                      "border-green-300 dark:border-green-700 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 font-medium";
                  } else if (isUserPick && !isCorrect) {
                    choiceClass +=
                      "border-red-300 dark:border-red-700 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 line-through";
                  } else {
                    choiceClass +=
                      "border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400";
                  }

                  return (
                    <div key={c.key} className={choiceClass}>
                      <span className="font-semibold mr-1">{c.key}.</span>
                      {c.text}
                      {isAnswer && (
                        <span className="ml-2 text-green-600 dark:text-green-400 font-bold">
                          ✓
                        </span>
                      )}
                      {isUserPick && !isCorrect && (
                        <span className="ml-2 text-red-500 font-bold">
                          ✗ (คำตอบของคุณ)
                        </span>
                      )}
                    </div>
                  );
                })}

                {q.explanation && (
                  <div className="mt-2 text-sm text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
                    <span className="font-semibold text-gray-700 dark:text-gray-300">
                      💡 อธิบาย:
                    </span>{" "}
                    {q.explanation}
                  </div>
                )}

                {!didAnswer && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 italic mt-1">
                    — ไม่ได้ตอบข้อนี้
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Print footer */}
      <div className="hidden print:block mt-8 pt-4 border-t-2 border-gray-900 text-center text-xs text-gray-500">
        <p>
          พิมพ์เมื่อ {dateStr} เวลา {timeStr} | วิชา {courseLabel} | {course.instructor}
        </p>
      </div>
    </div>
  );
}
