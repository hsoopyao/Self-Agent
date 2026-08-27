import os
import streamlit as st
import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

_ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages):
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        elif hasattr(msg, "content"):
            content = msg.content
        else:
            content = str(msg)
        total += len(_ENCODER.encode(content))
    return total

def summarize_messages(messages, llm):
    if not messages:
        return ""
    hist_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        hist_text += f"{role}: {content}\n"
    prompt = f"请用一段话（不超过250字）概括以下对话历史：\n{hist_text}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def trim_history(history, max_tokens=6000, target_ratio=0.6, max_rounds=3):
    """
    压缩历史，使总 Token 降至 max_tokens * target_ratio 以下。
    策略：从最早开始累积消息，直至累积 token 达到 batch_tokens（约1500），
          将这些消息压缩为一条摘要，然后递归检查。
    """
    if not history or len(history) <= 2:
        return history

    # 计算当前总 Token
    current_tokens = count_tokens(history)
    target_tokens = int(max_tokens * target_ratio)

    # 如果已经低于目标值，直接返回
    if current_tokens <= target_tokens:
        return history

    # 初始化 LLM（用于生成摘要）
    llm = ChatOpenAI(
        model=st.session_state.get("config_model_name", os.getenv("MODEL_NAME", "glm-4.7-flash")),
        api_key=os.getenv("BIGMODEL_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.3,
    )

    # 循环压缩，最多执行 max_rounds 轮
    for _ in range(max_rounds):
        if count_tokens(history) <= target_tokens:
            break

        # 如果历史消息太少，无法压缩（至少保留最近2条）
        if len(history) <= 4:
            break

        # 从最早开始累积消息，直到累积 token 达到 1500 或已包含至少 2 轮（4条）
        batch_messages = []
        batch_tokens = 0
        batch_size_limit = 1500  # 每次压缩大约 1500 tokens 的消息

        for msg in history:
            if batch_tokens >= batch_size_limit and len(batch_messages) >= 4:
                break
            batch_messages.append(msg)
            batch_tokens += count_tokens([msg])

        # 如果累积的消息少于 2 条，则不压缩（避免丢失信息）
        if len(batch_messages) < 4:
            break

        # 剩余消息
        remaining = history[len(batch_messages):]

        # 生成摘要
        summary_text = summarize_messages(batch_messages, llm)
        summary_msg = {"role": "assistant", "content": f"（历史摘要）{summary_text}"}

        # 重组历史
        history = [summary_msg] + remaining

    return history