# src/llm_client.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 在模块加载时加载环境变量
load_dotenv()

def get_llm(streaming: bool = True, temperature: float = 0.7, **kwargs):
    """
    获取 ChatOpenAI 实例，配置从环境变量和 session_state 读取。

    Args:
        streaming: 是否启用流式输出
        temperature: 温度参数（0~1）
        **kwargs: 其他 ChatOpenAI 参数

    Returns:
        ChatOpenAI 实例
    """
    api_key = os.getenv("BIGMODEL_API_KEY")
    if not api_key:
        raise ValueError("BIGMODEL_API_KEY 未设置，请检查 .env 文件")

    # 优先使用 session_state 中的模型名（设置页面可调），否则从环境变量读取，最终回退到默认值
    model_name = st.session_state.get(
        "config_model_name",
        os.getenv("MODEL_NAME", "glm-4.7-flash")
    )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=temperature,
        streaming=streaming,
        **kwargs
    )