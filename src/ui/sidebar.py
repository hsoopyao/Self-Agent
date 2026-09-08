import streamlit as st
import logging

from src.core.config import INTRODUCE
from src.core.context_manager import count_tokens
from src.retrieval.vectorstore import list_documents

logger = logging.getLogger(__name__)

def update_token_display():
    """更新 Token 显示（使用 session_state 中的容器）"""
    if "token_display" not in st.session_state:
        return
    threshold = st.session_state.config_max_tokens
    current_tokens = count_tokens(st.session_state.messages) if "messages" in st.session_state else 0
    status = "⚠️ 接近上限" if current_tokens > threshold * 0.8 else "✅ 正常"
    st.session_state.token_display.caption(
        f"📊 上下文 Token 数：**{current_tokens}** / {threshold} {status}"
    )

def render_sidebar():
    """构建并渲染侧边栏，返回 None。"""
    with st.sidebar:
        allow_web = st.toggle("🌐 允许联网", value=True, key="allow_web_switch")
        # 会话管理
        st.markdown("### 🧹 会话管理")
        if st.button("🗑️ 清空上下文窗口", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": INTRODUCE}]
            st.rerun()

        st.session_state.token_display = st.empty()
        update_token_display()
        st.divider()
        with st.expander("📚 知识库文档列表", expanded=False):
            docs = list_documents()
            if not docs:
                st.caption("暂无文档")
            else:
                # 按分类分组
                grouped = {}
                for doc in docs:
                    cat = doc.get("category", "未分类")
                    grouped.setdefault(cat, []).append(doc["filename"])
                for cat, files in grouped.items():
                    st.markdown(f"**{cat}** ({len(files)})")
                    for fname in files:
                        st.caption(f"• {fname}")
        st.divider()
        # 配置展示
        with st.expander("⚙️ 当前配置"):
            st.markdown(f"**模型**: `{st.session_state.config_model_name}`")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("全局阈值", f"{st.session_state.config_score_threshold:.2f}")
                st.metric("最大Token", st.session_state.config_max_tokens)
            with col2:
                st.metric("压缩比例", f"{st.session_state.config_target_ratio:.2f}")
                st.metric("ReAct最大步数", st.session_state.config_react_max_steps)
            st.caption(f"**触发React关键词**：{st.session_state.config_complex_keywords}")

        # 最后将 update_token_display 暴露给外部（以便在 chat_page 完成后调用）
        return update_token_display
