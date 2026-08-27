import streamlit as st
import time

from src.core.config import DEFAULT_CONFIG

st.set_page_config(page_title="设置", layout="centered")
st.title("⚙️ 设置")

# ---- 表单 ----
with st.form("settings_form"):
    st.markdown("### 📊 向量检索阈值")
    col1, col2 = st.columns(2)
    with col1:
        score_threshold = st.number_input(
            "全局 RAG 检索阈值",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config_score_threshold,
            step=0.05,
            help="内部知识库检索时，相似度高于此值才视为匹配"
        )
    with col2:
        temp_threshold = st.number_input(
            "临时文件检索阈值",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config_temp_score_threshold,
            step=0.05,
            help="会话临时文件检索时，相似度高于此值才视为匹配"
        )

    st.markdown("### 🧠 Token 上下文管理")
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.number_input(
            "最大上下文 Token 数",
            min_value=1000,
            max_value=20000,
            value=st.session_state.config_max_tokens,
            step=500,
            help="超过此值将触发历史压缩"
        )
    with col2:
        target_ratio = st.number_input(
            "压缩目标比例",
            min_value=0.3,
            max_value=0.9,
            value=st.session_state.config_target_ratio,
            step=0.05,
            help="压缩后 Token 数降至 `最大Token * 此比例`"
        )

    st.markdown("### 🔁 ReAct 触发关键词")
    complex_keywords = st.text_area(
        "输入触发复杂推理的关键词（英文逗号分隔）",
        value=st.session_state.config_complex_keywords,
        help="当问题中包含这些关键词时，自动启用 ReAct 多步推理"
    )

    st.markdown("### 💻 模型选择")
    model_name = st.text_input(
        "模型名称",
        value=st.session_state.config_model_name,
        help="支持智谱 BigModel 系列，如 glm-4.7-flash, glm-4, glm-5 等"
    )

    # 提交按钮
    submitted = st.form_submit_button("💾 保存设置")
    if submitted:
        st.session_state.config_score_threshold = score_threshold
        st.session_state.config_temp_score_threshold = temp_threshold
        st.session_state.config_max_tokens = max_tokens
        st.session_state.config_target_ratio = target_ratio
        st.session_state.config_complex_keywords = complex_keywords
        st.session_state.config_model_name = model_name
        st.toast("设置已保存！", icon="✅")
        st.balloons()
        time.sleep(2.0)
        st.rerun()

# ---- 重置为默认 ----
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 重置为默认设置"):
        for key, value in DEFAULT_CONFIG.items():
            st.session_state[key] = value
        st.toast("已重置为默认设置", icon="🔄")
        time.sleep(0.8)
        st.rerun()