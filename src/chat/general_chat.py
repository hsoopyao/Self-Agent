"""
通用对话模块：处理非内部知识的问题，支持联网搜索（流式）。
"""
import logging
import os
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from tavily.errors import TimeoutError as TavilyTimeoutError

from src.core.llm_client import get_llm
from src.core.config import load_prompt

GENERAL_SYSTEM = load_prompt("general_system.txt")
GENERAL_USER_TEMPLATE = load_prompt("general_user.txt")
logger = logging.getLogger(__name__)


def _get_search_timeout() -> float:
    try:
        timeout = float(os.getenv("TAVILY_SEARCH_TIMEOUT", "5"))
        return timeout if timeout > 0 else 5.0
    except (TypeError, ValueError):
        return 5.0


@lru_cache(maxsize=1)
def get_tavily_client():
    """延迟创建搜索客户端，避免未配置 Tavily 时阻断应用启动。"""
    return TavilyClient()


def search_results(query: str, timeout: float | None = None) -> list[dict]:
    """执行带 HTTP 请求超时的 Tavily 搜索并返回结果列表。"""
    response = get_tavily_client().search(
        query=query,
        max_results=5,
        include_answer=True,
        timeout=timeout if timeout is not None else _get_search_timeout(),
    )
    return response.get("results", [])

# 为了保持代码兼容性，可以定义一个统一的搜索接口
def search(query: str) -> str:
    results = search_results(query)
    return "\n".join([f"{item['title']}: {item['content']}" for item in results])


def search_with_timeout(query, timeout=None):
    """
    使用 Tavily 客户端的底层 HTTP 超时执行搜索。
    如果超时或失败，返回 None。
    """
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

def general_chat_stream(question: str, history: list = None):
    try:
        # 延迟到实际对话时创建，避免未配置 API Key 时阻断应用启动。
        llm = get_llm(streaming=True, temperature=0.7)
        search_result = search_with_timeout(question)
        search_info = f"搜索到的信息：{search_result}" if search_result else ""
        messages = [
            SystemMessage(content=GENERAL_SYSTEM),
        ]

        # 4. 如果有历史，将历史消息转为 LangChain 格式（只取最近 N 轮，避免超长）
        if history:
            # 取最近 10 条（即5轮对话），防止超出模型上下文窗口
            recent_history = history[-10:] if len(history) > 10 else history
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                # 如果有 system 或其他，忽略或适当处理

        # 5. 添加当前用户问题（带上搜索信息）
        current_user_content = GENERAL_USER_TEMPLATE.format(question=question, search_info=search_info)
        messages.append(HumanMessage(content=current_user_content))

        has_chunk = False
        for chunk in llm.stream(messages):
            has_chunk = True
            yield chunk.content
        if not has_chunk:
            yield "（未能生成回答，请稍后重试）"
    except Exception:
        logger.exception("联网聊天处理失败")
        # 用户友好提示
        yield "⚠️ 处理请求时出现内部错误，请稍后重试。"
