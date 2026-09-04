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

def route_query(question: str) -> str:
    """返回 'rag' / 'web' / 'chat'，通过规则优先匹配，未命中再调用 LLM。"""
    q_lower = question.lower().strip()

    # ---------- 规则匹配（高频场景） ----------
    # 1. 电影/排片/影院
    movie_keywords = ["排片", "影院", "电影院", "电影", "场次", "猫眼", "上映"]
    if any(kw in q_lower for kw in movie_keywords):
        return "web"   # 后续会触发 ReAct（见 chat_page 中的 need_react 逻辑）

    # 2. 天气
    weather_keywords = ["天气", "温度", "预报", "下雨", "晴", "多云", "气温"]
    if any(kw in q_lower for kw in weather_keywords):
        return "web"

    # 3. 内部知识库（文档、政策等）
    rag_keywords = ["文档", "政策", "公司", "内部", "规定", "制度", "手册", "说明"]
    if any(kw in q_lower for kw in rag_keywords):
        return "rag"

    # 4. 闲聊/问候（可简单判断，或直接走 chat 兜底）
    chat_keywords = ["你好", "介绍", "你是谁", "功能", "能力"]
    if any(kw in q_lower for kw in chat_keywords):
        return "chat"

    # ---------- 未命中规则，调用 LLM 路由 ----------
    # 原有逻辑保持不变
    llm = get_llm(streaming=False, temperature=0.1)
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=question)
    ]
    response = llm_with_tools.invoke(messages)
    logger.debug(f"response.tool_calls: {response.tool_calls}")
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        if tool_name == "rag_search":
            return "rag"
        elif tool_name == "web_search":
            return "web"
        elif tool_name == "direct_chat":
            return "chat"
    # fallback
    return "chat"