import re
import logging
import streamlit as st
from src.retrieval.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


@st.cache_resource
def get_task_index(collection_count: int) -> dict:
    """
    构建"任务编号 → 完整标题"的映射。
    从向量库的 heading_path_str 里提取。
    """
    vs = get_vectorstore()
    data = vs._collection.get(include=["documents", "metadatas"])

    index = {}   # {task_id: full_title}
    for meta in data["metadatas"]:
        heading = meta.get("heading_path_str", "")
        # 找 "3.1.1.1.3 XXX SIPOC 說明" 这种模式
        m = re.search(r"(\d+(?:\.\d+){2,})\s+([^>]+?)(?:\s*SIPOC)?\s*$", heading)
        if m:
            task_id = m.group(1)
            title = m.group(2).strip()
            if task_id not in index:
                index[task_id] = title

    logger.info(f"[任务索引] 共 {len(index)} 个任务")
    return index


def expand_query_with_task_index(query: str) -> str:
    """
    把查询里的任务编号展开成完整标题。
    例："3.1.1.1.3的执行角色" → "3.1.1.1.3 廠區審核需求 SIPOC 說明 執行角色"
    """
    from src.retrieval.vectorstore import get_vectorstore
    vs = get_vectorstore()
    total = vs._collection.count()
    index = get_task_index(total)

    if not index:
        return query

    # 提取查询里的所有编号
    task_ids = re.findall(r"\d+(?:\.\d+){2,}", query)
    if not task_ids:
        return query

    expanded_parts = [query]
    for tid in task_ids:
        # 精确匹配
        if tid in index:
            expanded_parts.append(f"{tid} {index[tid]}")
            continue
        # 前缀匹配：3.1.1.1 可能对应 3.1.1.1.1 或 3.1.1.1.3
        for k, v in index.items():
            if k.startswith(tid + ".") or k == tid:
                expanded_parts.append(f"{tid} {v}")
                break

    expanded = " ".join(expanded_parts)
    if expanded != query:
        logger.info(f"[编号展开] {query!r} → {expanded!r}")
    return expanded