import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from src.core.llm_client import get_llm
from src.core.config import load_prompt

logger = logging.getLogger(__name__)

@tool
def rag_search(query: str) -> str:
    """当用户询问公司内部文档、技术规范、PDF内容时调用此工具"""
    return "rag"

@tool
def web_search(query: str) -> str:
    """当用户需要实时信息、新闻、最新事件、股票、天气时调用此工具"""
    return "web"

@tool
def direct_chat(query: str) -> str:
    """"当用户闲聊、问候、询问助手自身信息时调用此工具，无需任务外部信息"""
    return "chat"

tools = [rag_search, web_search, direct_chat]

# 路由提示词 加载 Prompt
ROUTER_SYSTEM = load_prompt("router_system.txt")

def route_query(question: str, history: list | None = None) -> tuple[str, str]:
    """返回 (resolved_query, intent)。
    - resolved_query：结合 history 补全后的独立查询
    - intent：'rag' / 'web' / 'chat'
    """
    q_lower = question.lower().strip()

    # ---------- 规则匹配（只对"本身已完整"的问题生效） ----------
    # 注意：省略式追问几乎不会命中这些关键词，自然会落到 LLM 分支
    movie_keywords = ["排片", "影院", "电影院", "电影", "场次", "猫眼", "上映"]
    if any(kw in q_lower for kw in movie_keywords):
        return question, "web"

    weather_keywords = ["天气", "温度", "预报", "下雨", "晴", "多云", "气温"]
    if any(kw in q_lower for kw in weather_keywords):
        return question, "web"

    rag_keywords = ["文档", "政策", "公司", "内部", "规定", "制度", "手册", "说明"]
    if any(kw in q_lower for kw in rag_keywords):
        return question, "rag"

    chat_keywords = ["你好", "介绍", "你是谁", "功能", "能力"]
    if any(kw in q_lower for kw in chat_keywords):
        return question, "chat"

    # ---------- LLM 路由（带 history） ----------
    llm = get_llm(streaming=False, temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)

    messages = [SystemMessage(content=ROUTER_SYSTEM)]
    if history:
        # history 可以是 [{"role": "user", "content": ...}, ...]
        # 也可以是 LangChain Message 列表，直接 extend 即可
        messages.extend(history[-6:])
    messages.append(HumanMessage(content=question))

    response = llm_with_tools.invoke(messages)
    logger.debug(f"response.tool_calls: {response.tool_calls}")

    if response.tool_calls:
        tc = response.tool_calls[0]
        tool_name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        # 关键点：工具参数里的 query 就是 LLM 结合历史补全后的查询
        resolved_query = (args.get("query") or "").strip() or question

        mapping = {"rag_search": "rag", "web_search": "web", "direct_chat": "chat"}
        return resolved_query, mapping.get(tool_name, "chat")

    # fallback
    return question, "chat"