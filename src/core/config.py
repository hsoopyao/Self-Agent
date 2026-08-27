# src/config.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

INTRODUCE = "您好！我可以回答内部知识，也能进行常识问答和联网搜索。请问有什么可以帮助您？"

DEFAULT_CONFIG = {
    "config_score_threshold": 0.5,
    "config_temp_score_threshold": 0.3,
    "config_max_tokens": 6000,
    "config_target_ratio": 0.6,
    "config_complex_keywords": "对比,分析,为什么,总结,评价,区别,影响,解析,解读,解释,研究",
    "config_model_name": os.getenv("MODEL_NAME", "glm-4.7-flash"),
    "github_page": "https://github.com/hsoopyao/Self-Agent",
}

def init_config():
    """初始化 session_state 中的配置，如果未设置则使用默认值。"""
    for key, default_val in DEFAULT_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

# 确定项目根目录,读取 prompt 文件
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_prompt(filename):
    path = os.path.join(PROJECT_ROOT, "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()