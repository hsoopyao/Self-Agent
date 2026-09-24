import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

try:
    import opencc

    _cc_s2t = opencc.OpenCC('s2t')
except ImportError:
    _cc_s2t = None


def to_traditional(text: str) -> str:
    if _cc_s2t:
        return _cc_s2t.convert(text)
    return text


def _normalize(s: str) -> str:
    """去掉空白和标点，只留中英文数字"""
    return re.sub(r'[\s`*_\\|\[\]（）()：:；;，,。.、\-—/？?！!]', '', s)


# ---------- 只过滤真正的疑问词，不动可能作为章节名的词 ----------
_QUESTION_PHRASES = [
    "是什么", "是什麼", "是什麽", "是甚麼",
    "有什么", "有什麼",
    "有哪些", "有那些",
    "讲了什么", "讲了什麼", "讲了甚麼",
    "说了什么", "說了什麼", "說了甚麼",
    "怎么样", "怎麼樣", "咋样",
    "如何", "多少",
    "的区别", "有何区别", "有哪些区别",
    "吗", "嗎", "呢",
]

# ---------- 疑问词残留（用于 2 字词过滤） ----------
_QUESTION_WORDS = {
    "什麽", "什麼", "甚麼", "什么",
    "是什", "是甚", "有什", "有甚",
    "么呢", "麼呢",
    "哪個", "哪个", "哪些", "那些",
    "怎樣", "怎样", "如何",
    "多少", "幾個", "几个",
}


def extract_topic_keywords(query: str) -> list:
    """
    从查询提取主题词。
    - 只去掉疑问句式，不去掉"文檔/信息/需求"这类可能组成章节名的词
    - 加繁简双版本 + 2 字词拆分
    """
    q = query

    # 1. 去掉疑问句式
    for phrase in _QUESTION_PHRASES:
        q = q.replace(phrase, ' ')

    # 2. 去掉章节号前缀
    q = re.sub(r'\b\d+\.\s*', '', q)
    q = re.sub(r'\s+', ' ', q).strip()

    # 3. 提取 2 字以上的词
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', q)

    # 4. 加繁简双版本 + 2 字词拆分
    expanded = set()
    for kw in keywords:
        if len(kw) < 2:
            continue
        expanded.add(kw)
        expanded.add(to_traditional(kw))
        for i in range(len(kw) - 1):
            bigram = kw[i:i + 2]
            expanded.add(bigram)
            expanded.add(to_traditional(bigram))

    # 5. 加代码型词（NFR-01、FR-06 等）
    codes = re.findall(r'[A-Z]{2,}[-\s]?\d+', query, re.IGNORECASE)
    for c in codes:
        expanded.add(c.upper())
        expanded.add(c.upper().replace(' ', '-'))

    # 6. 过滤疑问词残留
    result = []
    seen = set()
    for kw in expanded:
        if not kw or kw in seen:
            continue
        if len(kw) < 2:
            continue
        if kw in _QUESTION_WORDS:
            continue
        if len(kw) == 2 and any(c in kw for c in "什麽甚麼"):
            continue
        seen.add(kw)
        result.append(kw)

    return result


def _extract_section_names_from_heading(heading_path_str: str) -> list:
    """从 heading_path_str 提取所有章节名（去掉编号前缀）"""
    names = []
    for part in heading_path_str.split(" > "):
        # 去编号前缀
        name = re.sub(r'^\d+\.\d*(?:\.\d+)*\s*', '', part).strip()
        name = re.sub(r'^\d+\.\s*', '', name).strip()
        name = name.strip('*').strip()
        if len(name) >= 2:
            names.append(name)
    return names


def search_by_heading_keywords(query: str, top_k: int = 8) -> list:
    """
    按 heading 关键词匹配章节。
    两级策略：
    1. 章节名精确匹配（章节名出现在 query 中）—— 最精确
    2. 关键词命中累积 —— 兜底
    """
    from src.retrieval.vectorstore import get_vectorstore

    vs = get_vectorstore()
    data = vs._collection.get(include=["documents", "metadatas"])

    query_norm = _normalize(query)
    query_trad = _normalize(to_traditional(query))

    # ==================================================
    # ---------- 1. 章节名精确匹配 ----------
    # ==================================================
    exact_matches = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        heading = meta.get("heading_path_str", "")
        section_names = _extract_section_names_from_heading(heading)

        max_match_len = 0
        for name in section_names:
            name_norm = _normalize(name)
            if len(name_norm) < 2:
                continue
            # 章节名出现在 query 里（简繁都试）
            if name_norm in query_norm or name_norm in query_trad:
                max_match_len = max(max_match_len, len(name_norm))
            # 也试繁体版本
            name_trad = _normalize(to_traditional(name))
            if name_trad != name_norm and (name_trad in query_norm or name_trad in query_trad):
                max_match_len = max(max_match_len, len(name_trad))

        if max_match_len > 0:
            exact_matches.append((max_match_len, doc, meta))

    if exact_matches:
        # 匹配长度降序（长的章节名优先，如"文檔信息"优先于"信息"）
        exact_matches.sort(key=lambda x: (-x[0], len(x[2].get("heading_path_str", ""))))
        result = [Document(page_content=doc, metadata=meta)
                  for _, doc, meta in exact_matches[:top_k]]
        logger.info(f"[heading-search] 章节名精确匹配 {len(result)} 条，"
                    f"top1: {result[0].metadata.get('heading_path_str')!r}")
        return result

    # ==================================================
    # ---------- 2. 回退到关键词匹配 ----------
    # ==================================================
    keywords = extract_topic_keywords(query)
    if not keywords:
        logger.info(f"[heading-search] 无关键词，跳过")
        return []

    logger.info(f"[heading-search] 关键词: {keywords[:12]}")

    scored = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        heading = meta.get("heading_path_str", "")
        body = doc
        heading_hits = sum(1 for kw in keywords if kw in heading)
        body_hits = sum(1 for kw in keywords if kw in body)
        total = heading_hits * 3 + body_hits
        if heading_hits >= 1 or body_hits >= 2:
            scored.append((total, heading_hits, doc, meta))

    scored.sort(key=lambda x: (-x[0], len(x[3].get("heading_path_str", ""))))

    result = [Document(page_content=doc, metadata=meta)
              for _, _, doc, meta in scored[:top_k]]

    if result:
        logger.info(f"[heading-search] 关键词匹配 {len(result)} 条，"
                    f"top1: {result[0].metadata.get('heading_path_str')!r}")
    else:
        logger.info(f"[heading-search] 无匹配")
    return result


def merge_docs(docs_a: list, docs_b: list) -> list:
    """合并去重"""
    seen = set()
    result = []
    for d in docs_a + docs_b:
        key = (d.metadata.get("page"), d.metadata.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result