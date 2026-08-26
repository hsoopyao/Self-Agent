# src/direct_chat.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(
    model=st.session_state.get("config_model_name", os.getenv("MODEL_NAME", "glm-4.7-flash")),
    api_key=os.getenv("BIGMODEL_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7,
    streaming=True,
)

DIRECT_SYSTEM = "你是一个友好的个人助手，名字叫卡卡西。请用口语化、亲切的语气回答用户的闲聊和一般性问题，无需使用外部信息。"

def direct_chat_stream(question: str, history: list = None):
    """
    纯聊天（无搜索、无RAG），支持多轮对话。
    :param question: 当前用户问题
    :param history: 历史消息列表，每个元素是 {"role": "user"/"assistant", "content": "..."}
    """
    # 构建消息列表，系统提示放在最前面
    messages = [
        SystemMessage(content=DIRECT_SYSTEM),
    ]

    # 如果有历史，将历史消息转换为 LangChain 消息对象（只取最近 N 轮，避免超长）
    if history:
        # 保留最近 10 条（即 5 轮对话），可根据需要调整
        recent_history = history[-10:] if len(history) > 10 else history
        for msg in recent_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            # 如果有 system 或其他角色，可忽略或按需处理

    # 添加当前用户问题
    messages.append(HumanMessage(content=question))

    # 流式生成
    try:
        for chunk in llm.stream(messages):
            yield chunk.content
    except Exception as e:
        yield f"⚠️ 生成回答时出错：{str(e)}"

# 同步版本，非流式
def direct_chat_sync(question: str, history: list = None) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    import os
    llm = ChatOpenAI(
        model=st.session_state.get("config_model_name", os.getenv("MODEL_NAME", "glm-4.7-flash")),
        api_key=os.getenv("BIGMODEL_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.7,
    )
    messages = [SystemMessage(content="你是一个友好的个人助手，名字叫卡卡西。")]
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content