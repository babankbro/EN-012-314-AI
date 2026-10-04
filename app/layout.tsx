import type { Metadata } from "next";
import { Prompt, Source_Code_Pro } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";
import AppShell from "@/components/AppShell";
import { course } from "@/lib/course";

const promptFont = Prompt({
  subsets: ["thai", "latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-prompt",
});

const sourceCodePro = Source_Code_Pro({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-code",
});

export const metadata: Metadata = {
  title: `${course.titleTh} — ${course.part} | ${course.code}`,
  description:
    "สื่อการสอนออนไลน์รายวิชาปัญญาประดิษฐ์ ภาค Machine Learning 9 บทเรียน พร้อมใบงาน แบบทบทวน และแบบทดสอบ",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="th" className={`${promptFont.variable} ${sourceCodePro.variable}`} suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var saved = localStorage.getItem('theme');
                  var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  if (saved === 'dark' || (!saved && prefersDark)) {
                    document.documentElement.classList.add('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="font-prompt bg-white dark:bg-gray-950 text-gray-800 dark:text-gray-200 antialiased">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
