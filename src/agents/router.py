import re
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

def is_rag_query(question: str) -> bool:
    q_lower = question.lower()
    # 1. 硬编码关键词（可扩展，建议从配置文件加载）
    rag_keywords = [
        "文档", "政策", "制度", "手册", "说明", "文件", "资料",
        "项目", "需求", "规格", "报告", "流程", "指南", "标准",
        "规范", "条款", "版本", "变更", "记录", "方案", "计划",
        "总结", "分析", "设计", "架构", "代码", "测试", "部署",
        "运维", "内部", "公司", "规定", "部门", "岗位", "职责"
    ]
    if any(kw in q_lower for kw in rag_keywords):
        return True

    # 2. 正则模式：版本号、章节编号、专业术语
    patterns = [
        r'v?\d+\.\d+(\.\d+)*',        # 版本号如 v3.1.1.1
        r'第[一二三四五六七八九十百千万]+[章节]',  # 第X章/节
        r'\d+\.\d+\.\d+',             # 纯数字版本如 3.1.1
        r'[A-Z]{2,}[-\s]?\d+',        # 业务代码如 PR-123
        r'需求|規格|spec|requirement', # 中英文需求词
    ]
    for pat in patterns:
        if re.search(pat, question, re.IGNORECASE):
            return True

    # 3. 判断问题长度（较长的技术性提问倾向知识库）
    if len(question.strip()) > 30:
        return True

    # 4. 默认为 False（即非 RAG）
    return False

def route_query(question: str) -> str:
    """返回 'rag' / 'web' / 'chat'，通过规则优先匹配，未命中再调用 LLM。"""
    q_lower = question.lower().strip()

    # ---------- 规则匹配（高频场景） ----------

    # 天气
    weather_keywords = ["天气", "温度", "预报", "下雨", "晴", "多云", "气温"]
    if any(kw in q_lower for kw in weather_keywords):
        return "web"

    # 内部知识库（文档、政策等）
    if is_rag_query:
        return "rag"

    # 闲聊/问候（可简单判断，或直接走 chat 兜底）
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