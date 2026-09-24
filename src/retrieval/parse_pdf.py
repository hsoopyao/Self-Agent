import os
import re
import logging
import tempfile
from typing import List, Tuple

import pymupdf4llm
import pymupdf
from collections import Counter, defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------- 分块配置 ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LEN = 20
MAX_SECTION_LEN = 8000

# ---------- 正则 ----------
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

_NUMBERED_HEADER_RE = re.compile(
    r"^(?:\*\*)?(\d+\.\d+(?:\.\d+)*)\s+([A-Z\u4e00-\u9fa5][^\n。，；！？]{1,79}?)(?:\*\*)?$",
    re.MULTILINE
)

_TOC_LINE_RE = re.compile(r"\.{3,}\s*\d+\s*$")

_PAGE_MARKER_RE = re.compile(
    r"(第\s*\d+\s*頁|第\s*\d+\s*页|Page\s*\d+|\d+\s*/\s*\d+\s*[頁页])",
    re.IGNORECASE
)


# ====================================================================
# 通用工具函数
# ====================================================================
def _clean_heading_title(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\*+\s*", "", s)
    s = re.sub(r"\s*\*+$", "", s)
    return s.strip()


def _is_toc_line(line: str) -> bool:
    return bool(_TOC_LINE_RE.search(line.strip()))


def _is_page_marker(text: str) -> bool:
    return bool(_PAGE_MARKER_RE.search(text))


def _page_marker_ratio(text: str) -> float:
    matches = _PAGE_MARKER_RE.findall(text)
    if not matches:
        return 0.0
    marker_chars = sum(len(m) for m in matches)
    total = len(text.strip())
    return marker_chars / total if total > 0 else 0.0


def _normalize_chars(s: str) -> str:
    return re.sub(r"[\s`*_\\|\[\]（）()：:；;，,。.、\-—/]", "", s)



def _page_has_table(md_text: str, min_rows: int = 3) -> bool:
    table_lines = sum(1 for l in md_text.split("\n") if l.strip().startswith("|"))
    return table_lines >= min_rows

# ====================================================================
# 表格字段提取
# ====================================================================
def _extract_table_fields(content: str, max_fields: int = 15) -> list:
    fields = []
    lines = [l.strip() for l in content.split("\n") if l.strip().startswith("|")]
    if not lines:
        return []

    header_line = None
    for s in lines:
        if set(s) <= set("|-: "):
            continue
        header_line = s
        break
    if not header_line:
        return []

    cells = [c.strip().strip("`*").strip() for c in header_line.strip("|").split("|")]
    cells = [c for c in cells if c]
    if not cells:
        return []

    has_digit = any(c.isdigit() for c in cells)

    if has_digit or len(cells) <= 2:
        for c in cells:
            if c and not c.isdigit() and c.lower() not in ("col2", "col3", "col4", "col5"):
                if c not in fields:
                    fields.append(c)
        if len(cells) == 2:
            for line in lines:
                if set(line) <= set("|-: "):
                    continue
                row_cells = [c.strip().strip("`*").strip() for c in line.strip("|").split("|")]
                if row_cells:
                    key = row_cells[0]
                    if key and not key.isdigit() and key not in fields and len(key) < 20:
                        fields.append(key)
    else:
        for i in range(0, len(cells), 2):
            c = cells[i]
            if c and not c.isdigit() and c.lower() not in ("col2", "col3", "col4", "col5"):
                if c not in fields:
                    fields.append(c)

    return fields[:max_fields]


# ====================================================================
# 按标题切分
# ====================================================================
def _split_by_headers(full_text: str) -> List[Tuple[int, int, List[str], str]]:
    matches = []

    for m in _HEADER_RE.finditer(full_text):
        matches.append({
            "start": m.start(),
            "level": len(m.group(1)),
            "title": _clean_heading_title(m.group(2)),
        })

    for m in _NUMBERED_HEADER_RE.finditer(full_text):
        line = m.group(0)
        if _is_toc_line(line):
            continue
        if any(abs(m.start() - x["start"]) < 5 for x in matches):
            continue
        level = m.group(1).count(".") + 2
        title = _clean_heading_title(line)
        matches.append({
            "start": m.start(),
            "level": level,
            "title": title,
        })

    matches.sort(key=lambda x: x["start"])

    if not matches:
        return [(0, len(full_text), [], full_text)]

    sections = []

    if matches[0]["start"] > 0:
        sections.append((0, matches[0]["start"], [], full_text[:matches[0]["start"]]))

    stack = []
    for i, m in enumerate(matches):
        level = m["level"]
        title = m["title"]

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]

        start = m["start"]
        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(full_text)
        sections.append((start, end, heading_path, full_text[start:end]))

    return sections


def _mark_cross_references(text: str) -> str:
    if "【引用其他任务" in text:
        return text

    def repl(m):
        return f"【引用其他任务，非本任务内容】{m.group(0)}【引用结束】"

    text = re.sub(r'完工後觸發"[\d.]+\s+[^"]+"', repl, text)
    text = re.sub(r'參考"[^"]+_SA"', repl, text)
    return text


def _find_page(char_pos: int, page_offsets: List[Tuple[int, int, int]]):
    for start, end, page_num in page_offsets:
        if start <= char_pos < end:
            return page_num
    return page_offsets[-1][2] if page_offsets else None


def split_keep_tables(text: str):
    def _is_table_line(line: str) -> bool:
        s = line.strip()
        return bool(s) and s.startswith("|")

    lines = text.split("\n")
    blocks = []
    buf = []
    in_table = False
    for line in lines:
        is_table = _is_table_line(line)
        if is_table and not in_table:
            if buf:
                blocks.append(("text", "\n".join(buf)))
                buf = []
            in_table = True
        elif not is_table and in_table:
            blocks.append(("table", "\n".join(buf)))
            buf = []
            in_table = False
        buf.append(line)
    if buf:
        blocks.append(("table" if in_table else "text", "\n".join(buf)))
    return blocks


def _merge_split_tables(blocks: list) -> list:
    merged = []
    i = 0
    while i < len(blocks):
        btype, content = blocks[i]
        if btype == "table":
            j = i + 1
            while j < len(blocks):
                next_type, next_content = blocks[j]
                if next_type == "table":
                    content += "\n" + next_content
                    j += 1
                elif next_type == "text" and not next_content.strip():
                    j += 1
                else:
                    break
            merged.append(("table", content))
            i = j
        else:
            merged.append(blocks[i])
            i += 1
    return merged


# ====================================================================
# 正文起始页检测
# ====================================================================
def detect_body_start_page(pages, min_chars=200):
    for i, p in enumerate(pages):
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if not lines:
            continue

        toc_lines = sum(1 for l in lines if _is_toc_line(l))
        if toc_lines >= 3:
            continue

        table_lines = sum(1 for l in lines if l.startswith("|"))
        table_ratio = table_lines / len(lines)
        dot_lines = sum(1 for l in lines if l.count(".") >= 5)
        char_count = sum(len(l) for l in lines if not l.startswith("|"))

        if table_ratio > 0.3:
            continue
        if dot_lines >= 3:
            continue
        if char_count < min_chars:
            continue

        return i

    return 0


# ====================================================================
# 用底层 PyMuPDF 补全 md
# ====================================================================
def _supplement_page_md(page_md: str, page_num: int, doc) -> str:
    if page_num > doc.page_count:
        return page_md

    page = doc[page_num - 1]
    raw_text = page.get_text()

    md_norm = _normalize_chars(page_md)

    raw_lines = raw_text.split("\n")
    missing_units = []
    i = 0
    while i < len(raw_lines):
        unit_lines = []
        while i < len(raw_lines) and len(unit_lines) < 5:
            line = raw_lines[i].strip()
            if line:
                unit_lines.append(line)
                i += 1
            else:
                if unit_lines:
                    break
                i += 1

        if not unit_lines:
            i += 1
            continue

        unit = " ".join(unit_lines)
        unit_norm = _normalize_chars(unit)

        # 太短，忽略
        if len(unit_norm) < 6:
            continue

        # 归一化精确匹配（这个是安全的）
        if unit_norm in md_norm:
            continue

        # ---------- 只保留"页眉页脚"这一种最安全的过滤 ----------
        # 其他一律不跳，宁可重复不可丢失
        if _is_page_marker(unit):
            matches = _PAGE_MARKER_RE.findall(unit)
            if len(matches) >= 2:
                continue
            if _page_marker_ratio(unit) >= 0.10:
                continue

        missing_units.append(unit)

    if not missing_units:
        return page_md

    supplement = "\n".join(missing_units)

    idx = page_md.find("####")
    if idx > 0:
        page_md = page_md[:idx] + "\n" + supplement + "\n\n" + page_md[idx:]
    else:
        page_md = page_md.rstrip() + "\n\n" + supplement

    return page_md


# ====================================================================
# 同页字符集子集去重
# ====================================================================
def _is_heading_chunk(content: str) -> bool:
    stripped = content.strip()
    if len(stripped) > 100:
        return False
    if re.search(r'\b\d+\.\d+(?:\.\d+)*\b', stripped):
        return True
    return False


def _remove_duplicate_chunks(chunks: list) -> list:
    by_page = defaultdict(list)
    for c in chunks:
        by_page[c.metadata.get("page")].append(c)

    result = []
    for page, page_chunks in by_page.items():
        page_chunks.sort(key=lambda c: -len(c.page_content))
        kept = []
        for c in page_chunks:
            if _is_heading_chunk(c.page_content):
                kept.append(c)
                continue

            c_chars = set(_normalize_chars(c.page_content))
            if not c_chars:
                kept.append(c)
                continue

            is_redundant = False
            for k in kept:
                if _is_heading_chunk(k.page_content):
                    continue
                k_chars = set(_normalize_chars(k.page_content))
                if c_chars.issubset(k_chars):
                    overlap = len(c_chars & k_chars) / len(c_chars)
                    if overlap >= 0.90:
                        is_redundant = True
                        break
            if not is_redundant:
                kept.append(c)
        result.extend(kept)
    return result


# ====================================================================
# 主解析函数
# ====================================================================
def parse_pdf_to_chunks(pdf_path, filename, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    top_y, bottom_y = detect_edge_y_bounds(pdf_path)
    logger.info(f"{filename}：页眉底={top_y}，页脚顶={bottom_y}")

    tmp_cropped = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    _crop_by_y(pdf_path, tmp_cropped, top_y, bottom_y)

    try:
        pages = pymupdf4llm.to_markdown(tmp_cropped, page_chunks=True)
    finally:
        os.unlink(tmp_cropped)

    if not pages:
        return []

    body_start = detect_body_start_page(pages)
    if body_start > 0:
        logger.info(f"{filename}：跳过前 {body_start} 页（封面/目录/说明）")
        pages = pages[body_start:]

    raw_doc = pymupdf.open(pdf_path)
    try:
        for p in pages:
            page_num = p["metadata"].get("page", 1)
            p["text"] = _supplement_page_md(p["text"], page_num, raw_doc)
    finally:
        raw_doc.close()

    full_text = ""
    page_offsets = []
    for idx, p in enumerate(pages):
        start = len(full_text)
        full_text += p["text"].rstrip() + "\n\n"
        end = len(full_text)
        meta = p.get("metadata", {}) or {}
        page_num = meta.get("page") or meta.get("page_number") or (idx + 1)
        page_offsets.append((start, end, int(page_num)))

    sections = _split_by_headers(full_text)

    doc_title = None
    if sections:
        first_page = _find_page(sections[0][0], page_offsets)
        for s, e, hp, _ in sections:
            if hp and len(hp) == 1 and _find_page(s, page_offsets) == first_page:
                doc_title = hp[0]
                break

    def strip_doc_title(hp):
        return hp[1:] if doc_title and hp and hp[0] == doc_title else hp

    def is_noise(hp, content):
        last = hp[-1].lower() if hp else ""
        if "contents" in last or "目录" in last or "目錄" in last:
            return True

        stripped = content.strip()

        # 表格内容豁免（优先）
        if stripped.startswith("|"):
            return False

        # 目录行
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        toc_lines = sum(1 for l in lines if _is_toc_line(l))
        if toc_lines >= 2:
            return True
        if content.count("....") >= 3:
            return True

        # 页眉页脚残留（只判多次出现或高占比）
        if _is_page_marker(stripped):
            matches = _PAGE_MARKER_RE.findall(stripped)
            if len(matches) >= 2:
                return True
            if _page_marker_ratio(stripped) >= 0.10:
                return True

        if len(stripped) < MIN_CHUNK_LEN:
            return True
        return False

    sections = [
        (s, e, strip_doc_title(hp), t)
        for s, e, hp, t in sections
        if not is_noise(strip_doc_title(hp), t)
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        keep_separator=True,
    )

    chunks = []
    cursor = 0
    for sec_start, sec_end, heading_path, sec_text in sections:
        if len(sec_text.strip()) < MIN_CHUNK_LEN:
            continue

        sec_text_marked = _mark_cross_references(sec_text)

        blocks = split_keep_tables(sec_text_marked)
        blocks = _merge_split_tables(blocks)

        for block_type, block in blocks:
            block = block.strip()
            if len(block) < MIN_CHUNK_LEN:
                continue
            if block_type == "table" or len(block) <= chunk_size:
                pieces = [block]
            else:
                pieces = text_splitter.split_text(block)

            for sub in pieces:
                sub = sub.strip()
                if len(sub) < MIN_CHUNK_LEN:
                    continue

                pos = full_text.find(sub, cursor)
                if pos == -1:
                    orig_sub = sub.replace("【引用其他任务，非本任务内容】", "").replace("【引用结束】", "")
                    pos = full_text.find(orig_sub, cursor)
                if pos == -1:
                    pos = sec_start
                else:
                    cursor = max(cursor, pos)
                page_num = _find_page(pos, page_offsets)

                if sub.startswith("|"):
                    fields = _extract_table_fields(sub)
                    prefix_parts = []
                    if heading_path:
                        prefix_parts.append("章节：" + " > ".join(heading_path))
                    prefix_parts.append("内容类型：表格")
                    if fields:
                        prefix_parts.append("字段：" + "、".join(fields))
                    sub = f"【{'；'.join(prefix_parts)}】\n{sub}"

                chunks.append(Document(
                    page_content=sub,
                    metadata={
                        "filename": filename,
                        "doc_title": doc_title or filename,
                        "page": page_num,
                        "heading_path": heading_path,
                        "heading_path_str": " > ".join(heading_path),
                        "chunk_index": len(chunks),
                    },
                ))

    before = len(chunks)
    chunks = _remove_duplicate_chunks(chunks)
    after = len(chunks)
    if before != after:
        logger.info(f"{filename}：去重 {before} → {after} chunk")

    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i

    logger.info(f"{filename}：{len(pages)} 页 → {len(sections)} 章节 → {len(chunks)} chunk")
    return chunks


def parse_pdf_bytes_to_chunks(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return parse_pdf_to_chunks(tmp_path, filename, chunk_size, chunk_overlap)
    finally:
        os.unlink(tmp_path)


# ====================================================================
# 页眉页脚检测
# ====================================================================
def detect_edge_y_bounds(pdf_path: str, ratio_threshold=0.5, edge_ratio=0.15):
    doc = pymupdf.open(pdf_path)
    n_pages = doc.page_count
    if n_pages < 3:
        doc.close()
        return None, None

    line_page_count = Counter()
    line_y_ranges = {}

    for page in doc:
        h = page.rect.height
        seen = set()
        for block in page.get_text("blocks"):
            _, y0, _, y1, text = block[:5]
            in_top = y1 < h * edge_ratio
            in_bottom = y0 > h * (1 - edge_ratio)
            if not (in_top or in_bottom):
                continue
            for line in text.split("\n"):
                line = line.strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                line_page_count[line] += 1
                if line in line_y_ranges:
                    ly0, ly1 = line_y_ranges[line]
                    line_y_ranges[line] = (min(ly0, y0), max(ly1, y1))
                else:
                    line_y_ranges[line] = (y0, y1)

    doc.close()

    threshold = max(2, int(n_pages * ratio_threshold))
    repeated = [l for l, c in line_page_count.items() if c >= threshold]
    if not repeated:
        return None, None

    doc = pymupdf.open(pdf_path)
    mid = doc[0].rect.height / 2
    doc.close()

    top_y1s = [line_y_ranges[l][1] for l in repeated if line_y_ranges[l][1] < mid]
    bottom_y0s = [line_y_ranges[l][0] for l in repeated if line_y_ranges[l][0] > mid]

    top_y = max(top_y1s) if top_y1s else None
    bottom_y = min(bottom_y0s) if bottom_y0s else None
    return top_y, bottom_y


def _crop_by_y(pdf_path: str, output_path: str, top_y=None, bottom_y=None, margin=2):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        r = page.rect
        y0 = (top_y + margin) if top_y is not None else r.y0
        y1 = (bottom_y - margin) if bottom_y is not None else r.y1
        if y1 <= y0:
            continue
        page.set_cropbox(pymupdf.Rect(r.x0, y0, r.x1, y1))
    doc.save(output_path)
    doc.close()