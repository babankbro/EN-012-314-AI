"use client";

import { useEffect, useState } from "react";

let renderCount = 0;

/** แสดงแผนภาพ Mermaid (โหลด library เฉพาะหน้าที่มีแผนภาพ) */
export default function Mermaid({ chart }: { chart: string }) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const isDark = document.documentElement.classList.contains("dark");
    import("mermaid").then(async ({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        theme: isDark ? "dark" : "default",
        fontFamily: "var(--font-prompt), sans-serif",
        securityLevel: "strict",
      });
      try {
        // id ต้องไม่ซ้ำทุกครั้งที่ render (React strict mode เรียก effect สองรอบ)
        const { svg } = await mermaid.render(`mermaid-${++renderCount}`, chart);
        if (!cancelled) setSvg(svg);
      } catch {
        if (!cancelled) setError(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [chart]);

  if (error) {
    return (
      <pre className="my-5 p-4 rounded-xl bg-gray-50 dark:bg-gray-900 text-xs overflow-x-auto border border-gray-200 dark:border-gray-800">
        {chart}
      </pre>
    );
  }

  return (
    <div
      className="my-6 flex justify-center overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-4 min-h-[80px] [&_svg]:max-w-full [&_svg]:h-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
