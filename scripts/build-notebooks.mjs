// แปลงใบงาน content/labs/lab-XX.md → content/notebooks/lab-XX.ipynb
// บล็อก ```python กลายเป็น code cell ส่วนที่เหลือเป็น markdown cell
// รัน: npm run notebooks
import { readdir, readFile, writeFile, mkdir } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const LABS_DIR = path.join(ROOT, "content", "labs");
const OUT_DIR = path.join(ROOT, "content", "notebooks");

const HEADER = (title) =>
  `# ${title}\n\n` +
  `**EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence)** — ผศ.ดร. สรายุทธ กรวิรัตน์  \n` +
  `สาขาวิชาวิศวกรรมคอมพิวเตอร์ มหาวิทยาลัยกาฬสินธุ์\n\n` +
  `ชื่อ-สกุล: ______________________  รหัสนักศึกษา: ______________`;

function toLines(text) {
  const lines = text.split("\n");
  return lines.map((line, i) => (i < lines.length - 1 ? line + "\n" : line));
}

function markdownCell(text) {
  return { cell_type: "markdown", metadata: {}, source: toLines(text) };
}

function codeCell(text) {
  return { cell_type: "code", execution_count: null, metadata: {}, outputs: [], source: toLines(text) };
}

function splitCells(markdown) {
  const cells = [];
  const fence = /^```(\w*)[^\n]*\n([\s\S]*?)^```[ \t]*$/gm;
  let last = 0;
  let match;
  const pushMarkdown = (text) => {
    const trimmed = text.trim();
    if (trimmed) cells.push(markdownCell(trimmed));
  };
  while ((match = fence.exec(markdown)) !== null) {
    const [whole, lang, body] = match;
    if (lang === "python" || lang === "py") {
      pushMarkdown(markdown.slice(last, match.index));
      cells.push(codeCell(body.replace(/\n$/, "")));
      last = match.index + whole.length;
    }
  }
  pushMarkdown(markdown.slice(last));
  return cells;
}

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const files = (await readdir(LABS_DIR)).filter((f) => /^lab-\d+\.md$/.test(f)).sort();
  for (const file of files) {
    const id = file.match(/\d+/)[0];
    const markdown = await readFile(path.join(LABS_DIR, file), "utf-8");
    const firstHeading = markdown.match(/^\|\s*ชื่อใบงาน\s*\|\s*(.+?)\s*\|/m)?.[1] ?? `ใบงานที่ ${id}`;
    const notebook = {
      cells: [markdownCell(HEADER(firstHeading)), ...splitCells(markdown)],
      metadata: {
        kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
        language_info: { name: "python" },
        colab: { provenance: [] },
      },
      nbformat: 4,
      nbformat_minor: 5,
    };
    const out = path.join(OUT_DIR, `lab-${id}.ipynb`);
    await writeFile(out, JSON.stringify(notebook, null, 1) + "\n", "utf-8");
    const codeCells = notebook.cells.filter((c) => c.cell_type === "code").length;
    console.log(`lab-${id}.ipynb  (${notebook.cells.length} cells, ${codeCells} code)`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
