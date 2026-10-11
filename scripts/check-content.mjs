// ตรวจความครบถ้วนและรูปแบบของเนื้อหาใน content/ — รัน: npm run check
import { readFile, access } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CONTENT = path.join(ROOT, "content");
const IDS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"];
const KEYS = ["ก", "ข", "ค", "ง"];

const problems = [];
const report = (id, msg) => problems.push(`[${id}] ${msg}`);

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function checkMarkdown(id, file) {
  const full = path.join(CONTENT, file);
  if (!(await exists(full))) {
    report(id, `ไม่พบ ${file}`);
    return undefined;
  }
  const md = await readFile(full, "utf-8");
  const lines = md.split("\n").length;
  // ตัดบล็อกโค้ดและโค้ดในบรรทัดออกก่อนตรวจ (เช่น df[["a","b"]] ไม่ใช่ wikilink)
  const prose = md.replace(/^```[\s\S]*?^```/gm, "").replace(/`[^`\n]*`/g, "");

  if (/\[\[[^\]]+\]\]/.test(prose)) report(id, `${file}: ยังมี [[wikilink]] ของ Obsidian`);
  if (/^# /m.test(prose)) report(id, `${file}: มีหัวข้อ H1 (#) — หน้าเว็บแสดงชื่อบทให้แล้ว`);
  if ((md.match(/^```/gm) ?? []).length % 2 !== 0) report(id, `${file}: จำนวน \`\`\` ไม่ครบคู่`);

  for (const m of md.matchAll(/!\[[^\]]*\]\(([^)]+)\)/g)) {
    const src = m[1];
    if (src.startsWith("/images/") && !(await exists(path.join(ROOT, "public", src)))) {
      report(id, `${file}: ไม่พบรูป ${src}`);
    }
  }

  for (const block of md.matchAll(/^```mermaid\n([\s\S]*?)^```/gm)) {
    // ข้อความในโหนดที่มีวงเล็บต้องอยู่ในเครื่องหมายคำพูด ไม่งั้น mermaid parse ไม่ผ่าน
    const unquoted = block[1].replace(/"[^"\n]*"/g, '""');
    for (const node of unquoted.matchAll(/\w+\s*[\[\(\{]+([^"\]\)\}][^\]\}]*)[\]\)\}]+/g)) {
      if (/[()]/.test(node[1])) report(id, `${file}: mermaid โหนด "${node[1].slice(0, 40)}" ควรใส่ "..."`);
    }
  }
  return lines;
}

async function checkExam(id) {
  const file = `exams/exam-${id}.json`;
  const full = path.join(CONTENT, file);
  if (!(await exists(full))) {
    report(id, `ไม่พบ ${file}`);
    return undefined;
  }
  let exam;
  try {
    exam = JSON.parse(await readFile(full, "utf-8"));
  } catch (e) {
    report(id, `${file}: JSON ไม่ถูกต้อง (${e.message})`);
    return undefined;
  }
  if (exam.moduleId !== id) report(id, `${file}: moduleId = ${exam.moduleId}`);
  const qs = exam.questions ?? [];
  if (qs.length !== 20) report(id, `${file}: มี ${qs.length} ข้อ (ควรเป็น 20)`);
  const dist = Object.fromEntries(KEYS.map((k) => [k, 0]));
  qs.forEach((q, i) => {
    if (q.id !== i + 1) report(id, `${file}: ข้อที่ ${i + 1} มี id = ${q.id}`);
    const keys = (q.choices ?? []).map((c) => c.key).join("");
    if (keys !== KEYS.join("")) report(id, `${file}: ข้อ ${q.id} ตัวเลือก = ${keys}`);
    if (!KEYS.includes(q.answer)) report(id, `${file}: ข้อ ${q.id} คำตอบ = ${q.answer}`);
    else dist[q.answer]++;
    if (!q.explanation?.trim()) report(id, `${file}: ข้อ ${q.id} ไม่มีคำอธิบาย`);
  });
  return KEYS.map((k) => `${k}${dist[k]}`).join(" ");
}

async function main() {
  console.log("บท | module | lab | review | exam (การกระจายคำตอบ)");
  for (const id of IDS) {
    const mod = await checkMarkdown(id, `modules/module-${id}.md`);
    const lab = await checkMarkdown(id, `labs/lab-${id}.md`);
    const sheet = path.join(CONTENT, "SHEET", `review_sheet_${id}.html`);
    const hasSheet = await exists(sheet);
    if (!hasSheet) report(id, `ไม่พบ SHEET/review_sheet_${id}.html`);
    else if (!(await readFile(sheet, "utf-8")).includes('href="sheet.css"')) report(id, `review_sheet_${id}.html ไม่ได้ใช้ sheet.css`);
    const exam = await checkExam(id);
    console.log(`${id} | ${mod ?? "-"} บรรทัด | ${lab ?? "-"} บรรทัด | ${hasSheet ? "✓" : "-"} | ${exam ?? "-"}`);
  }
  if (problems.length) {
    console.log(`\nพบ ${problems.length} ปัญหา:`);
    problems.forEach((p) => console.log("  - " + p));
    process.exitCode = 1;
  } else {
    console.log("\nเนื้อหาครบและรูปแบบถูกต้อง");
  }
}

main();
