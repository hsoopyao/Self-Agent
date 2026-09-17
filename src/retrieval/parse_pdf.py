import os
import re
import logging
import tempfile
from typing import List, Tuple

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------- 分块配置 ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_LEN = 20          # 过短的 chunk 丢弃（通常是空标题）
MAX_SECTION_LEN = 8000      # 单个 section 超过这个长度才二次切分

# Markdown 标题正则：匹配行首的 # ## ### ...
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

def _extract_table_fields(content: str, max_fields: int = 12) -> list:
    """从 Markdown 表格里提取字段名（前 4 行，排除分隔行和占位符）"""
    fields = []
    for line in content.split("\n")[:4]:
        s = line.strip()
        if not s.startswith("|"):
            continue
        # 跳过 |---|---| 分隔行
        if set(s) <= set("|-: "):
            continue
        for cell in s.strip("|").split("|"):
            c = cell.strip().strip("`*").strip()
            if not c:
                continue
            if c.isdigit():
                continue
            if c.lower() in ("col2", "col3", "col4", "col5"):
                continue
            if c not in fields:
                fields.append(c)
    return fields[:max_fields]

def _split_by_headers(full_text: str) -> List[Tuple[int, int, List[str], str]]:
    """
    按 Markdown 标题切分全文，返回 [(start, end, heading_path, content), ...]
    heading_path 是从根到当前标题的路径，如 ['第三章', '3.2 登录模块']
    """
    matches = list(_HEADER_RE.finditer(full_text))
    sections = []

    if not matches:
        return [(0, len(full_text), [], full_text)]

    # 第一个标题之前的内容，归到"无标题"区
    if matches[0].start() > 0:
        sections.append((0, matches[0].start(), [], full_text[: matches[0].start()]))

    stack: List[Tuple[int, str]] = []   # [(level, title), ...]
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()

        # 弹出层级 >= 当前的，保持栈是严格的父子链
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        heading_path = [t for _, t in stack]

        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append((start, end, heading_path, full_text[start:end]))

    return sections


def _mark_cross_references(text: str) -> str:
    """
    把正文里"引用其他任务"的部分加标记，提示 LLM 不是归属切换。
    特征：如 "完工後觸發"3.5.1.1.2 T1B-PO..." 这种引用。
    """
    import re
    # 匹配 "觸發"XXX 任务名" 的模式
    # 简化：如果一段话里出现 "觸發" + 任务编号 + 任务名，加标记
    def repl(m):
        return f"【引用其他任务，非本任务内容】{m.group(0)}【引用结束】"

    # 匹配 "觸發"XXX T1B-PO..." 这种
    pattern = r'完工後觸發"[\d.]+\s+[^"]+"'
    text = re.sub(pattern, repl, text)

    # 匹配 "參考"..._SA" 这种
    pattern2 = r'參考"[^"]+_SA"'
    text = re.sub(pattern2, repl, text)

    return text

def _find_page(char_pos: int, page_offsets: List[Tuple[int, int, int]]):
    """根据字符位置反查页码"""
    for start, end, page_num in page_offsets:
        if start <= char_pos < end:
            return page_num
    return page_offsets[-1][2] if page_offsets else None

def split_keep_tables(text: str):
    """把 text 拆成 [(type, content), ...]，表格整体不切"""
    lines = text.split("\n")
    blocks = []
    buf = []
    in_table = False
    for line in lines:
        is_table_line = line.strip().startswith("|") and line.strip().endswith("|")
        if is_table_line and not in_table:
            # 表格开始，先冲刷之前的缓冲
            if buf:
                blocks.append(("text", "\n".join(buf)))
                buf = []
            in_table = True
        elif not is_table_line and in_table:
            # 表格结束
            blocks.append(("table", "\n".join(buf)))
            buf = []
            in_table = False
        buf.append(line)
    if buf:
        blocks.append(("table" if in_table else "text", "\n".join(buf)))
    return blocks


def detect_body_start_page(pages, min_chars=200):
    """
    找到第一页正文，返回其索引（0-based）。
    跳过封面 / 修订历史 / 批准 / 目录页。
    """
    for i, p in enumerate(pages):
        lines = [l.strip() for l in p["text"].split("\n") if l.strip()]
        if not lines:
            continue

        table_lines = sum(1 for l in lines if l.startswith("|"))
        table_ratio = table_lines / len(lines)
        dot_lines = sum(1 for l in lines if l.count(".") >= 5)
        # 排除表格行后剩余的文字量
        char_count = sum(len(l) for l in lines if not l.startswith("|"))

        # 表格为主的页 → 封面 / Revision History / Approvals
        if table_ratio > 0.3:
            continue
        # 大量省略号 → 目录
        if dot_lines >= 3:
            continue
        # 非表格文字太少 → 封面
        if char_count < min_chars:
            continue

        return i

    return 0

def _supplement_page_md(page_md: str, page_num: int, doc) -> str:
    """
    用底层 PyMuPDF 的文本补全 pymupdf4llm 丢失的内容。
    关键：用连续多行组成"单元"作为指纹，避免短词误判。
    """
    if page_num > doc.page_count:
        return page_md

    page = doc[page_num - 1]
    raw_text = page.get_text()

    def _norm(s: str) -> str:
        return re.sub(r"[\s`*_\\|\[\]（）()：:；;，,。.、]", "", s)

    md_norm = _norm(page_md)

    # 把底层文本按"连续非空行"组成单元，每个单元 2-5 行
    raw_lines = raw_text.split("\n")

    missing_units = []
    i = 0
    while i < len(raw_lines):
        # 收集连续非空行
        unit_lines = []
        while i < len(raw_lines) and len(unit_lines) < 5:
            line = raw_lines[i].strip()
            if line:
                unit_lines.append(line)
                i += 1
            else:
                # 空行是单元边界
                if unit_lines:
                    break
                i += 1

        if not unit_lines:
            i += 1
            continue

        unit = " ".join(unit_lines)

        # 跳过页眉页脚
        if any(kw in unit for kw in [
            "System Requirement Analysis",
            "DMP_Workplace_V3.1.1.1",
            "_SA_V1.0",
        ]):
            continue
        # 跳过纯页码
        if unit.strip().isdigit():
            continue

        unit_norm = _norm(unit)
        if len(unit_norm) < 6:
            continue

        # 用整个单元作为指纹
        if unit_norm not in md_norm:
            missing_units.append(unit)

    if not missing_units:
        return page_md

    supplement = "\n".join(missing_units)

    # 插到 #### 标题之前（保持续表归属上一节）
    idx = page_md.find("####")
    if idx > 0:
        page_md = page_md[:idx] + "\n" + supplement + "\n\n" + page_md[idx:]
    else:
        page_md = page_md.rstrip() + "\n\n" + supplement

    return page_md

def parse_pdf_to_chunks(pdf_path, filename, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # 自适应检测页眉页脚 y 边界
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

    # 拼接 + 页偏移
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

    # 剔除文档级标题
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
        # 只看当前章节自己的标题，避免父级 Contents 污染
        last = hp[-1].lower() if hp else ""
        if "contents" in last or "目录" in last or "目錄" in last:
            return True
        # 内容里大量省略号 → 目录页残留
        if content.count("....") >= 3:
            return True
        # 表格内容豁免
        stripped = content.strip()
        if stripped.startswith("|"):
            return False
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

        # 标记跨任务引用(【引用其他任务，非本任务内容】"3.5.1.1.2 T1B-PO 確認信息完整性與正確性"【引用结束】)
        sec_text = _mark_cross_references(sec_text)

        # 表格整体保留，非表格按大小切
        for block_type, block in split_keep_tables(sec_text):
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

                # ⚠️ 先反查位置（用原始文本），再加前缀
                pos = full_text.find(sub, cursor)
                if pos == -1:
                    pos = sec_start
                else:
                    cursor = max(cursor, pos)
                page_num = _find_page(pos, page_offsets)

                # 表格块：加语境前缀
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

    logger.info(f"{filename}：{len(pages)} 页 → {len(sections)} 章节 → {len(chunks)} chunk")
    return chunks


def parse_pdf_bytes_to_chunks(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """从字节流解析（供 Streamlit 上传用）"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return parse_pdf_to_chunks(tmp_path, filename, chunk_size, chunk_overlap)
    finally:
        os.unlink(tmp_path)

import pymupdf
from collections import Counter

def detect_edge_y_bounds(pdf_path: str, ratio_threshold=0.5, edge_ratio=0.15):
    """
    返回 (top_y, bottom_y)，表示页眉底边和页脚顶边的绝对 y 坐标。
    检测不到返回 (None, None)。
    """
    doc = pymupdf.open(pdf_path)
    n_pages = doc.page_count
    if n_pages < 3:
        doc.close()
        return None, None

    line_page_count = Counter()
    line_y_ranges = {}   # line -> (min_y0, max_y1)

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

    # 取页面高度的一半作为顶/底分界
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