"use client";

import { isValidElement, useState, type ReactElement, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import rehypeKatex from "rehype-katex";
import { Check, Copy, Info, Lightbulb, TriangleAlert } from "lucide-react";
import Mermaid from "./Mermaid";

interface MarkdownContentProps {
  content: string;
}

const LANG_LABELS: Record<string, string> = {
  python: "Python",
  py: "Python",
  bash: "Terminal",
  sh: "Terminal",
  text: "Text",
  json: "JSON",
  csv: "CSV",
};

const CALLOUTS = {
  note: { title: "หมายเหตุ", icon: Info },
  tip: { title: "เคล็ดลับ", icon: Lightbulb },
  warning: { title: "ข้อควรระวัง", icon: TriangleAlert },
} as const;

type CalloutKind = keyof typeof CALLOUTS;

/**
 * rehype plugin: เปลี่ยน blockquote ที่ขึ้นต้นด้วย [!NOTE] / [!TIP] / [!WARNING]
 * ให้เป็น callout (รูปแบบเดียวกับ GitHub / Obsidian)
 */
function rehypeCallouts() {
  const walk = (node: any) => {
    if (!node.children) return;
    for (const child of node.children) {
      if (child.type === "element" && child.tagName === "blockquote") {
        const p = child.children.find((c: any) => c.type === "element");
        const first = p?.tagName === "p" ? p.children[0] : undefined;
        const match = first?.type === "text" && /^\s*\[!(NOTE|TIP|WARNING)\][ \t]*\n?/i.exec(first.value);
        if (match) {
          first.value = first.value.slice(match[0].length);
          child.properties = { ...child.properties, dataCallout: match[1].toLowerCase() };
        }
      }
      walk(child);
    }
  };
  return (tree: any) => walk(tree);
}

function nodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join("");
  if (isValidElement(node)) return nodeText((node.props as { children?: ReactNode }).children);
  return "";
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);

  if (lang === "output") {
    return (
      <div className="output-block-wrapper">
        <div className="output-block-header">
          <span className="output-label">Output</span>
        </div>
        <pre className="output-block">
          <code>{code}</code>
        </pre>
      </div>
    );
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard ไม่พร้อมใช้งาน (เช่น http ที่ไม่ใช่ localhost) — ไม่ต้องทำอะไร
    }
  };

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-lang">{LANG_LABELS[lang] ?? (lang || "Code")}</span>
        <button type="button" onClick={copy} className="code-copy" aria-label="คัดลอกโค้ด">
          {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          {copied ? "คัดลอกแล้ว" : "คัดลอก"}
        </button>
      </div>
      <pre className="code-block">
        <code>{code}</code>
      </pre>
    </div>
  );
}

const components: Components = {
  h1: ({ children }) => (
    <h1 className="text-2xl font-bold text-gray-900 dark:text-white mt-8 mb-4 first:mt-0">{children}</h1>
  ),
  h2: ({ children, id }) => (
    <h2 id={id} className="text-xl font-bold text-gray-900 dark:text-white mt-10 mb-3 pb-2 border-b border-gray-200 dark:border-gray-700 scroll-mt-20">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100 mt-6 mb-2">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="font-semibold text-gray-800 dark:text-gray-100 mt-5 mb-2">{children}</h4>
  ),
  p: ({ children }) => <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">{children}</p>,
  ul: ({ children }) => (
    <ul className="list-disc space-y-1 mb-4 text-gray-700 dark:text-gray-300 pl-6">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal space-y-1 mb-4 text-gray-700 dark:text-gray-300 pl-6">{children}</ol>
  ),
  li: ({ children }) => <li className="leading-relaxed pl-1">{children}</li>,
  blockquote: ({ children, node }) => {
    const kind = node?.properties?.dataCallout as CalloutKind | undefined;
    if (kind && CALLOUTS[kind]) {
      const { title, icon: Icon } = CALLOUTS[kind];
      return (
        <div className={`callout callout-${kind}`}>
          <div className="callout-title">
            <Icon className="w-4 h-4" />
            {title}
          </div>
          {children}
        </div>
      );
    }
    return (
      <blockquote className="border-l-4 border-blue-400 dark:border-blue-600 pl-4 py-1 my-4 bg-blue-50 dark:bg-blue-900/20 rounded-r-lg text-sm text-gray-600 dark:text-gray-400">
        {children}
      </blockquote>
    );
  },
  table: ({ children }) => (
    <div className="overflow-x-auto mb-6">
      <table className="w-full text-sm border-collapse border border-gray-300 dark:border-gray-700">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-gray-100 dark:bg-gray-800">{children}</thead>,
  th: ({ children, style }) => (
    <th style={style} className="px-4 py-2 text-left font-semibold text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-700">
      {children}
    </th>
  ),
  td: ({ children, style }) => (
    <td style={style} className="px-4 py-2 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 align-top">
      {children}
    </td>
  ),
  tr: ({ children }) => <tr className="even:bg-gray-50 dark:even:bg-gray-900/30">{children}</tr>,
  pre: ({ children }) => {
    const codeEl = Array.isArray(children) ? children[0] : children;
    const className = isValidElement(codeEl)
      ? ((codeEl as ReactElement<{ className?: string }>).props.className ?? "")
      : "";
    const lang = /language-([\w-]+)/.exec(className)?.[1]?.toLowerCase() ?? "";
    const code = nodeText(codeEl).replace(/\n$/, "");
    if (lang === "mermaid") return <Mermaid chart={code} />;
    return <CodeBlock lang={lang} code={code} />;
  },
  code: ({ children }) => (
    <code className="bg-gray-100 dark:bg-gray-800 text-pink-600 dark:text-pink-400 px-1.5 py-0.5 rounded text-[0.9em] font-mono">
      {children}
    </code>
  ),
  // markdown วางรูปไว้ใน <p> จึงใช้ <span> แบบ block แทน <figure> เพื่อให้ HTML ถูกต้อง
  img: ({ src, alt }) => (
    <span className="block my-6">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={typeof src === "string" ? src : undefined}
        alt={alt ?? ""}
        loading="lazy"
        className="block mx-auto max-w-full h-auto rounded-xl border border-gray-200 dark:border-gray-800 bg-white"
      />
      {alt && <span className="block text-center text-xs text-gray-500 dark:text-gray-400 mt-2">{alt}</span>}
    </span>
  ),
  hr: () => <hr className="my-8 border-t border-gray-200 dark:border-gray-700" />,
  strong: ({ children }) => <strong className="font-semibold text-gray-900 dark:text-white">{children}</strong>,
  a: ({ href, children }) => {
    const external = href?.startsWith("http");
    return (
      <a
        href={href}
        target={external ? "_blank" : undefined}
        rel={external ? "noopener noreferrer" : undefined}
        className="text-blue-600 dark:text-blue-400 hover:underline"
      >
        {children}
      </a>
    );
  },
};

/**
 * remark-math ถือว่า `$$ ... $$` ที่อยู่บรรทัดเดียวเป็นสูตรในบรรทัด
 * จึงแยก `$$` ออกเป็นบรรทัดของตัวเองให้แสดงเป็นสูตรแยกบรรทัด (ข้ามบล็อกโค้ด)
 */
function normalizeDisplayMath(markdown: string): string {
  return markdown
    .split(/(^```[\s\S]*?^```)/m)
    .map((part, i) =>
      i % 2 === 1 ? part : part.replace(/^([ \t]*)\$\$(.+?)\$\$[ \t]*$/gm, "$1$$$$\n$1$2\n$1$$$$")
    )
    .join("");
}

export default function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeRaw, rehypeKatex, rehypeCallouts]}
      components={components}
    >
      {normalizeDisplayMath(content)}
    </ReactMarkdown>
  );
}
