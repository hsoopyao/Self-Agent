from src.core.config import load_prompt
from src.core.llm_client import get_llm
from src.retrieval.vectorstore import get_retriever

# 加载 Prompt
RAG_SYSTEM = load_prompt("rag_qa_prompt.md")
RAG_USER_TEMPLATE = """
参考资料：
{context}

用户问题：{question}
"""


def _format_chunk(d, idx: int) -> str:
    """每个 chunk 前加编号 [N]，不塞完整路径"""
    return f"[{idx}] {d.page_content}"


def _build_context(docs) -> str:
    """拼接 context，只用编号"""
    return "\n\n".join(
        [_format_chunk(d, i) for i, d in enumerate(docs, 1)]
    )


def _build_sources_footer(docs) -> str:
    if not docs:
        return ""

    by_file = {}
    for i, d in enumerate(docs, 1):
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
            # 用列表项 -，每行独立渲染
            lines.append(f"- {idx_str} {heading_safe}{page_str}")
    return "\n".join(lines)


def _format_page_range(pages: list) -> str:
    """把 [10, 11, 12] 压缩为 p.10-12；[10, 12] 显示为 p.10, 12"""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p.{pages[0]}"
    # 检查是否连续
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"p.{pages[0]}-{pages[-1]}"
    return "p." + ", ".join(str(p) for p in pages)


def rag_chain_stream(input_dict: dict):
    llm = get_llm(streaming=True, temperature=0.2)
    retriever = get_retriever(k=3)

    question = input_dict["input"]
    docs = retriever.invoke(question)
    context = _build_context(docs)

    messages = [
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE.format(context=context, question=question))
    ]

    for chunk in llm.stream(messages):
        yield chunk.content

    yield _build_sources_footer(docs)


def rag_chain_with_docs(docs, question: str):
    """
    基于已有的文档列表生成流式回答（不再次检索）
    """
    llm = get_llm(streaming=True, temperature=0.2)
    context = _build_context(docs)

    messages = [
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE.format(context=context, question=question))
    ]

    for chunk in llm.stream(messages):
        yield chunk.content

    yield _build_sources_footer(docs)