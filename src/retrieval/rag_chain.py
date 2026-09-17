import re
import logging
import time

from src.core.config import load_prompt
from src.core.llm_client import get_llm
from src.retrieval.vectorstore import get_retriever

logger = logging.getLogger(__name__)

# 加载 Prompt
RAG_SYSTEM = load_prompt("rag_qa_prompt.md")
RAG_USER_TEMPLATE = """
参考资料：
{context}

用户问题：{question}
"""


# ---------- 工具函数 ----------
def _format_chunk(d, idx: int) -> str:
    """每个 chunk 前加编号 [N]"""
    return f"[{idx}] {d.page_content}"


def _build_context(docs) -> str:
    """拼接 context，只用编号"""
    return "\n\n".join(
        [_format_chunk(d, i) for i, d in enumerate(docs, 1)]
    )


def _extract_cited_indices(text: str) -> set:
    """从正文里提取实际引用的 [N] 编号"""
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def _build_sources_footer(docs, cited: set) -> str:
    """只输出正文引用到的来源"""
    if not docs or not cited:
        return ""

    filtered = [(i, d) for i, d in enumerate(docs, 1) if i in cited]
    if not filtered:
        return ""

    by_file = {}
    for i, d in filtered:
        meta = d.metadata
        filename = meta.get("filename", "未知文件")
        heading = meta.get("heading_path_str", "无章节")
        page = meta.get("page")
        by_file.setdefault(filename, {})
        by_file[filename].setdefault((heading, page), []).append(i)

    lines = ["\n\n---\n**参考来源**"]
    for filename, items in by_file.items():
        lines.append(f"\n**{filename}**\n")
        for (heading, page), idxs in items.items():
            idx_str = "".join(f"[{n}]" for n in idxs)
            heading_safe = heading.replace(">", "›").replace("<", "‹")
            page_str = f" · p.{page}" if page is not None else ""
            lines.append(f"- {idx_str} {heading_safe}{page_str}")
    return "\n".join(lines)

# ---------- 主函数 ----------
def rag_chain_stream(input_dict: dict):
    """检索 + 生成（旧路径，保留兼容）"""
    llm = get_llm(streaming=True, temperature=0.2)
    retriever = get_retriever(k=3)

    question = input_dict["input"]
    docs = retriever.invoke(question)
    context = _build_context(docs)

    messages = [
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE.format(context=context, question=question)),
    ]

    # 收集完整输出
    collected = ""
    for chunk in llm.stream(messages):
        collected += chunk.content


    cited = _extract_cited_indices(collected)
    logger.info(f"[RAG 生成] 正文引用编号: {sorted(cited)}")

    yield collected
    yield _build_sources_footer(docs, cited)


def rag_chain_with_docs(docs, question: str):
    """基于已有的文档列表生成回答（不再次检索）"""
    llm = get_llm(streaming=True, temperature=0.2)
    context = _build_context(docs)

    logger.info(f"[RAG 生成] context 长度={len(context)} 字, docs 数={len(docs)}")

    messages = [
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE.format(context=context, question=question)),
    ]

    t0 = time.time()
    first_token_time = None
    collected = ""

    # ---------- 收集全部输出 ----------
    for chunk in llm.stream(messages):
        if first_token_time is None:
            first_token_time = time.time()
            logger.info(f"[RAG 生成] 首 token 延迟 {first_token_time - t0:.2f}s")
        collected += chunk.content

    t1 = time.time()
    logger.info(f"[RAG 生成] 总耗时 {t1 - t0:.2f}s, 原始 {len(collected)} 字")

    # ---------- 提取引用编号 ----------
    cited = _extract_cited_indices(collected)
    logger.info(f"[RAG 生成] 正文引用编号: {sorted(cited)}")

    # ---------- 输出 ----------
    yield collected
    yield _build_sources_footer(docs, cited)