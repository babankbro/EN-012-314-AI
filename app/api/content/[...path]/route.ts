import { NextRequest, NextResponse } from "next/server";
import { readFile, stat } from "fs/promises";
import path from "path";

const CONTENT_DIR = path.join(process.cwd(), "content");

const MIME_TYPES: Record<string, string> = {
  ".pdf": "application/pdf",
  ".html": "text/html; charset=utf-8",
  ".htm": "text/html; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".ipynb": "application/x-ipynb+json",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
};

// ไฟล์ที่ให้ดาวน์โหลดแทนการเปิดในเบราว์เซอร์
const DOWNLOAD_EXTS = new Set([".csv", ".ipynb"]);

export async function GET(
  _request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const resolved = path.resolve(CONTENT_DIR, ...params.path);

    // Security: prevent directory traversal
    if (!resolved.startsWith(CONTENT_DIR + path.sep)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const fileStat = await stat(resolved);
    if (!fileStat.isFile()) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    const buffer = await readFile(resolved);
    const ext = path.extname(resolved).toLowerCase();
    const filename = encodeURIComponent(path.basename(resolved));

    const headers: Record<string, string> = {
      "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
      "Content-Length": buffer.length.toString(),
    };
    if (ext === ".pdf") {
      headers["Content-Disposition"] = `inline; filename*=UTF-8''${filename}`;
    } else if (DOWNLOAD_EXTS.has(ext)) {
      headers["Content-Disposition"] = `attachment; filename*=UTF-8''${filename}`;
    }

    return new NextResponse(buffer, { status: 200, headers });
  } catch (error: any) {
    if (error.code === "ENOENT") {
      return NextResponse.json({ error: "File not found" }, { status: 404 });
    }
    console.error("Error serving file:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
