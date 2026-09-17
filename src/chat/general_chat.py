"""
通用对话模块：处理非内部知识的问题，支持联网搜索（流式）。
"""
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

from src.core.llm_client import get_llm
from src.core.config import load_prompt, get_temperature

GENERAL_SYSTEM = load_prompt("general_prompt.md")
GENERAL_USER_TEMPLATE = """
问题：{question}
搜索到的信息：{search_info}
"""
logger = logging.getLogger(__name__)


# ---------- 搜索客户端 ----------
def _get_search_timeout() -> float:
    try:
        timeout = float(os.getenv("TAVILY_SEARCH_TIMEOUT", "8"))
        return timeout if timeout > 0 else 8.0
    except (TypeError, ValueError):
        return 8.0


@lru_cache(maxsize=1)
def get_tavily_client():
    return TavilyClient()


def search_results(query: str, timeout: float | None = None) -> list[dict]:
    response = get_tavily_client().search(
        query=query,
        max_results=5,
        include_answer=True,
        timeout=timeout if timeout is not None else _get_search_timeout(),
    )
    return response.get("results", [])


def search(query: str) -> str:
    results = search_results(query)
    return "\n".join([f"{item['title']}: {item['content']}" for item in results])


def search_with_timeout(query, timeout=None):
    try:
        results = search_results(query, timeout=timeout)
        if not results:
            return None
        return "\n".join([f"{item['title']}: {item['content']}" for item in results])
    except TavilyTimeoutError:
        logger.warning("Tavily 搜索超时，query=%r", query)
        return None
    except Exception:
        logger.exception("Tavily 搜索失败，query=%r", query)
        return None


# ---------- 查询改写 ----------
def _rewrite_query_with_history(question: str, history: list) -> str:
    """用 LLM 把追问改写成独立查询（补全主题词）"""
    if not history:
        return question

    # 只在短追问时改写
    if len(question.strip()) > 30:
        return question

    # 取最近 2 轮对话
    recent = history[-4:]
    hist_text = "\n".join([
        f"{m.get('role', '?')}: {m.get('content', '')[:150]}"
        for m in recent
    ])

    prompt = f"""基于对话历史，把用户的追问改写成一条独立的搜索查询词。
只输出查询词，不要解释、不要引号、不要换行。

对话历史：
{hist_text}

用户追问：{question}

改写后的独立查询："""

    try:
        llm = get_llm(streaming=False, temperature=0.0)
        rewritten = llm.invoke(prompt).content.strip()
        # 清理引号、换行
        rewritten = rewritten.strip('"\'').split("\n")[0].strip()
        if not rewritten:
            return question
        logger.info(f"[查询改写] {question!r} → {rewritten!r}")
        return rewritten
    except Exception as e:
        logger.warning(f"查询改写失败: {e}")
        return question


def _append_date_to_query(query: str, now: datetime) -> str:
    """如果查询含时间词，拼上具体日期"""
    date_map = {
        "今天": now.strftime("%Y-%m-%d"),
        "今日": now.strftime("%Y-%m-%d"),
        "现在": now.strftime("%Y-%m-%d"),
        "最新": now.strftime("%Y-%m-%d"),
        "昨天": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "昨日": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        "明天": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        "明日": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
        "后天": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
    }
    for kw, date in date_map.items():
        if kw in query:
            # 已经含日期就不重复拼
            if date in query:
                return query
            return f"{query} {date}"
    return query


def _build_system_prompt(now: datetime) -> str:
    """构造带日期的 system prompt"""
    weekday = "一二三四五六日"[now.weekday()]
    today_str = now.strftime("%Y年%m月%d日")
    return (
        GENERAL_SYSTEM
        + f"\n\n【当前日期】{today_str}（星期{weekday}）"
        + "\n【铁律】"
        + f"\n1. 今天是 {today_str}，这是唯一正确的日期。"
        + "\n2. 涉及时间的回答一律以【当前日期】为准。"
        + "\n3. 搜索结果里如果出现其他日期（如2024年、2025年），那是网页旧信息，忽略。"
    )


# ---------- 主函数 ----------
def general_chat_stream(question: str, history: list = None):
    try:
        now = datetime.now()
        llm = get_llm(streaming=True, temperature=get_temperature("chat"))

        # ---------- 1. 构造带日期的 system prompt ----------
        system_prompt = _build_system_prompt(now)

        # ---------- 2. 追问改写 ----------
        search_query = question
        if history:
            search_query = _rewrite_query_with_history(question, history)

        # ---------- 3. 拼日期 ----------
        search_query = _append_date_to_query(search_query, now)

        logger.info(f"[general_chat] 原始: {question!r}")
        logger.info(f"[general_chat] 搜索词: {search_query!r}")

        # ---------- 4. 搜索（首次 + 失败重试） ----------
        search_result = search_with_timeout(search_query)

        if not search_result:
            # 简化关键词重试
            simple = " ".join(search_query.split()[:3])
            logger.warning(f"[general_chat] 首次搜索失败，简化重试: {simple!r}")
            search_result = search_with_timeout(simple)

        search_info = f"搜索到的信息：{search_result}" if search_result else "（未搜索到相关信息）"

        # ---------- 5. 组装消息 ----------
        messages = [SystemMessage(content=system_prompt)]

        if history:
            recent_history = history[-10:] if len(history) > 10 else history
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        current_user_content = GENERAL_USER_TEMPLATE.format(
            question=question, search_info=search_info
        )
        messages.append(HumanMessage(content=current_user_content))

        # ---------- 6. 流式输出 ----------
        has_chunk = False
        for chunk in llm.stream(messages):
            has_chunk = True
            yield chunk.content
        if not has_chunk:
            yield "（未能生成回答，请稍后重试）"

    except Exception:
        logger.exception("联网聊天处理失败")
        yield "⚠️ 处理请求时出现内部错误，请稍后重试。"