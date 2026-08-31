from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from src.core.llm_client import get_llm
from src.core.config import load_prompt

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
    """返回 'rag' / 'web' """
    llm = get_llm(streaming=False, temperature=0.1)
    # 绑定工具
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=question)
    ]
    response = llm_with_tools.invoke(messages)
    # 提取工具调用
    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        # 映射到我们的分类
        if tool_name == "rag_search":
            return "rag"
        elif tool_name == "web_search":
            return "web"
        elif tool_name == "direct_chat":
            return "chat"
    # fallback：如果模型未调用工具，默认用直接搜索
    return "chat"