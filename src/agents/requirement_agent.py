import re
import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Generator

from src.core.config import load_prompt, get_temperature
from src.core.llm_client import get_llm
from src.retrieval.vectorstore import get_vectorstore
from src.core.structured_store import save_requirements

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CACHE_DIR = Path(PROJECT_ROOT, "./db/analysis_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR = Path(PROJECT_ROOT, "./db/analysis_history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

JSON_MARKER = "===REQUIREMENTS_JSON==="

# ---------- 缓存 key ----------
def _cache_key(filename: str) -> str:
    """文档名 + prompt hash + 模型名 组成缓存 key"""
    prompt_text = load_prompt("requirement_analyst.md")
    prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
    model = os.getenv("LLM_MODEL", "default")
    raw = f"{filename}|{prompt_hash}|{model}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------- 主入口 ----------
def requirement_analysis_stream(
    filename: str,
    max_chars: int = 30000,
    use_cache: bool = True,
) -> Generator[str, None, None]:
    """
    读全文 → 流式生成需求分析报告 → 落库。
    use_cache=True 命中缓存直接返回，False 强制重新分析。
    """
    cache_key = _cache_key(filename)
    cache_path = CACHE_DIR / f"{cache_key}.json"

    # ---------- 缓存命中 ----------
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            report = cached.get("report", "")
            logger.info(f"缓存命中：{filename}")
            # 模拟流式（每 30 字一块）
            for i in range(0, len(report), 30):
                yield report[i:i + 30]
            items = cached.get("items", [])
            if items:
                save_requirements(filename, items)
            return
        except Exception as e:
            logger.warning(f"缓存读取失败，重新分析: {e}")

    # ---------- 正常分析 ----------
    chunks = _load_doc_chunks(filename)
    if not chunks:
        yield f"未找到文档《{filename}》的解析内容，请先上传。"
        return

    full_input = _format_for_llm(chunks)
    if len(full_input) > max_chars:
        full_input = full_input[:max_chars]

    prompt_text = load_prompt("requirement_analyst.md")
    llm = get_llm(streaming=True, temperature=get_temperature("requirement"))
    messages = [
        ("system", prompt_text),
        ("human", f"请分析以下需求文档内容：\n\n{full_input}")
    ]

    collected = []
    for chunk in llm.stream(messages):
        collected.append(chunk.content)
        yield chunk.content

    report = "".join(collected)

    # ---------- 抽取需求条目 ----------
    items = _extract_req_items(report, filename)

    # ---------- 写缓存 + 历史版本 ----------
    try:
        cache_path.write_text(
            json.dumps({"report": report, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _save_history(filename, items, report)
    except Exception as e:
        logger.warning(f"缓存/历史写入失败: {e}")

    if items:
        save_requirements(filename, items)


# ---------- 历史版本 ----------
def _save_history(filename: str, items: list, report: str):
    """保存一份历史版本，只保留最近 10 份"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\u4e00-\u9fff.-]", "_", filename)
    path = HISTORY_DIR / f"{safe}_{ts}.json"
    path.write_text(
        json.dumps(
            {"timestamp": ts, "filename": filename, "items": items, "report": report},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    # 清理旧版本
    prefix = safe + "_"
    versions = sorted(
        HISTORY_DIR.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in versions[10:]:
        try:
            old.unlink()
        except Exception:
            pass


def list_history_versions(filename: str) -> list:
    """列出某文档的所有历史版本（新→旧）"""
    safe = re.sub(r"[^\w\u4e00-\u9fff.-]", "_", filename)
    prefix = safe + "_"
    versions = sorted(
        HISTORY_DIR.glob(f"{prefix}*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for p in versions:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            result.append({
                "path": str(p),
                "timestamp": data.get("timestamp", ""),
                "count": len(data.get("items", [])),
            })
        except Exception:
            continue
    return result


def load_history_version(path: str) -> dict:
    """读取某历史版本"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------- 内部函数 ----------
def _load_doc_chunks(filename: str):
    vs = get_vectorstore()
    data = vs._collection.get(include=["documents", "metadatas"])
    chunks = [
        {"content": doc, "meta": meta}
        for doc, meta in zip(data["documents"], data["metadatas"])
        if meta.get("filename") == filename
    ]
    chunks.sort(key=lambda x: (
        x["meta"].get("page", 0),
        x["meta"].get("chunk_index", 0),
    ))
    return chunks


def _format_for_llm(chunks: list) -> str:
    parts = []
    for c in chunks:
        heading = c["meta"].get("heading_path_str", "（无章节）")
        page = c["meta"].get("page", "?")
        parts.append(f"### [{heading}] (p.{page})\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def _extract_req_items(report: str, filename: str) -> list:
    """按 JSON 分隔符抽取需求条目"""
    _, _, json_part = report.partition(JSON_MARKER)
    if not json_part.strip():
        matches = re.findall(r"```json\s*(.*?)```", report, re.DOTALL | re.IGNORECASE)
        if matches:
            json_part = matches[-1]
        else:
            logger.warning("未找到 JSON 分隔符，需求条目为空")
            return []

    text = re.sub(r"^```(?:json)?\s*", "", json_part.strip())
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    result = []
    seen = set()
    for it in data:
        if not isinstance(it, dict):
            continue
        req_id = str(it.get("req_id", "")).strip().upper()
        if not req_id.startswith("REQ-") or req_id in seen:
            continue
        seen.add(req_id)
        result.append({
            "req_id": req_id,
            "title": str(it.get("title", ""))[:200],
            "description": str(it.get("description", ""))[:1000],
            "acceptance": str(it.get("acceptance", ""))[:1000],
            "source": str(it.get("source", ""))[:200],
            "priority": str(it.get("priority", ""))[:10],
        })
    result.sort(key=lambda x: x["req_id"])
    return result