"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Menu, Moon, Sun, Brain, Search, X, ExternalLink } from "lucide-react";
import { modules, type ModuleData } from "@/lib/modules-data";
import { course } from "@/lib/course";

interface HeaderProps {
  onMenuToggle: () => void;
}

function matchModules(query: string): ModuleData[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return modules.filter(
    (m) =>
      m.titleTh.toLowerCase().includes(q) ||
      m.titleEn.toLowerCase().includes(q) ||
      m.description.toLowerCase().includes(q) ||
      m.topics.some((t) => t.toLowerCase().includes(q)) ||
      m.id.includes(q)
  );
}

export default function Header({ onMenuToggle }: HeaderProps) {
  const [isDark, setIsDark] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const router = useRouter();
  const searchResults = matchModules(searchQuery);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggleTheme() {
    const next = !isDark;
    setIsDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // storage ถูกปิด — ใช้ธีมได้เฉพาะรอบนี้
    }
  }

  function closeSearch() {
    setSearchOpen(false);
    setSearchQuery("");
  }

  function handleResultClick(id: string) {
    closeSearch();
    router.push(`/module/${id}`);
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
        document.getElementById("site-search")?.focus();
      }
      if (e.key === "Escape") closeSearch();
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  const resultList = (
    <>
      {searchResults.map((mod) => (
        <button
          key={mod.id}
          onClick={() => handleResultClick(mod.id)}
          className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
        >
          <span className="text-xs font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-2 py-0.5 rounded">
            {mod.id}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-900 dark:text-white truncate">{mod.titleTh}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{mod.titleEn}</p>
          </div>
        </button>
      ))}
    </>
  );

  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-14 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800 print:hidden">
      <div className="flex items-center justify-between h-full px-4">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuToggle}
            className="lg:hidden p-2 -ml-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Toggle menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <Link href="/" className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-gradient-to-br from-violet-600 to-blue-600 rounded-lg flex items-center justify-center shadow-sm">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div className="hidden sm:block leading-tight">
              <span className="block text-sm font-semibold text-gray-900 dark:text-white">
                {course.titleEn}
              </span>
              <span className="block text-[11px] text-gray-500 dark:text-gray-400">
                {course.code} · {course.part}
              </span>
            </div>
          </Link>
        </div>

        <div className="flex-1 max-w-md mx-4 hidden md:block">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              id="site-search"
              type="text"
              placeholder="ค้นหาบทเรียน หรือหัวข้อ... (Ctrl+K)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setSearchOpen(true)}
              onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
              className="w-full pl-9 pr-4 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
            {searchOpen && searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50">
                {resultList}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setSearchOpen(!searchOpen)}
            className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Search"
          >
            <Search className="w-5 h-5" />
          </button>

          <a
            href="https://colab.research.google.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg text-orange-700 dark:text-orange-300 bg-orange-50 dark:bg-orange-900/30 hover:bg-orange-100 dark:hover:bg-orange-900/50 transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            <span>Google Colab</span>
          </a>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Toggle theme"
          >
            {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {searchOpen && (
        <div className="md:hidden border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-3">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="ค้นหาบทเรียน..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus
              className="w-full pl-9 pr-10 py-2.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
            />
            <button
              onClick={closeSearch}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700"
              aria-label="Close search"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
          {searchResults.length > 0 && (
            <div className="mt-2 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              {resultList}
            </div>
          )}
        </div>
      )}
    </header>
  );
}
