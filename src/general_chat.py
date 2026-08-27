"""
通用对话模块：处理非内部知识的问题，支持联网搜索（流式）。
"""
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.llm_client import get_llm

# 加载 Prompt
def load_prompt(filename):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read().strip()

GENERAL_SYSTEM = load_prompt("general_system.txt")
GENERAL_USER_TEMPLATE = load_prompt("general_user.txt")

# 1. 初始化 LLM（流式）
llm = get_llm(streaming=True, temperature=0.7)

# 2. 初始化 Tavily 搜索引擎
tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    # search_depth="advanced", # 如需深度搜索可启用
)

# 为了保持代码兼容性，可以定义一个统一的搜索接口
def search(query: str) -> str:
    try:
        # Tavily的invoke方法返回一个列表，我们提取内容
        results = tavily_tool.invoke({"query": query})
        # 将结果格式化为字符串
        return "\n".join([f"{item['title']}: {item['content']}" for item in results])
    except Exception as e:
        raise e

def search_with_timeout(query, timeout=2):
    """2秒超时的搜索，超时返回 None"""
    try:
        results = tavily_tool.invoke({"query": query})
        if not results:
            return None
        return "\n".join([f"{item['title']}: {item['content']}" for item in results])
    except Exception:
        return None  # 任何异常都视为搜索失败

def general_chat_stream(question: str, history: list = None):
    try:
        search_result = search_with_timeout(question, timeout=2)
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
    except Exception as e:
        yield f"⚠️ 生成回答时出错：{str(e)}"