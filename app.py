import os
import sys

import streamlit as st


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(page_title="智能助手", layout="centered")

from src.context_manager import count_tokens, trim_history
from src.direct_chat import direct_chat_stream
from src.general_chat import general_chat_stream
from src.pages.knowledges import main as knowledge_page
from src.rag_chain import rag_chain_with_docs
from src.router import route_query
from src.theme import GITHUB_URL, apply_theme, get_theme, initialize_theme, toggle_theme
from src.vectorstore import (
    chunk_file_from_bytes,
    create_temp_vectorstore,
    get_vectorstore,
    list_documents,
    search_with_score,
)


def render_thought(content: str) -> str:
    """渲染思考过程。"""
    return f'<div class="react-card react-thought">{content}</div>'

def render_action(content: str) -> str:
    """渲染工具调用。"""
    return f'<div class="react-card react-action">{content}</div>'

def render_observation(content: str) -> str:
    """渲染观察结果和可点击出处。"""
    return f'<div class="react-card react-observation">{content}</div>'

INTRODUCE = "您好！我可以回答内部知识，也能进行常识问答和联网搜索。请问有什么可以帮助您？"

initialize_theme()
apply_theme()

# 初始化配置（如果 session_state 中没有）
DEFAULT_CONFIG = {
    "config_score_threshold": 0.5,
    "config_temp_score_threshold": 0.3,
    "config_max_tokens": 6000,
    "config_target_ratio": 0.6,
    "config_complex_keywords": "对比,分析,为什么,总结,评价,比较,区别,影响,解析,解读,解释,研究",
    "config_model_name": st.session_state.get("config_model_name", os.getenv("MODEL_NAME", "glm-4.7-flash")),
}
for key, default_val in DEFAULT_CONFIG.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# 启动时初始化向量库（后台加载）
get_vectorstore()

# ========== 全局侧边栏（所有页面共享） ==========
with st.sidebar:
    st.markdown("### 🎨 外观与项目")
    theme_col, github_col = st.columns(2)
    with theme_col:
        theme_label = "🌙 深色" if get_theme() == "light" else "☀️ 浅色"
        if st.button(theme_label, use_container_width=True, key="theme_toggle"):
            toggle_theme()
            st.rerun()
    with github_col:
        st.link_button("⭐ GitHub", GITHUB_URL, use_container_width=True)

    st.divider()
    st.markdown("### 🧹 会话管理")
    if st.button("🗑️ 清空上下文窗口", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": INTRODUCE}
        ]
        st.rerun()

    # ---------- Token 实时显示 ----------
    token_display = st.empty()
    threshold = st.session_state.config_max_tokens

    def update_token_display():
        """更新 Token 显示（内部函数，复用 caption 逻辑）"""
        if "messages" in st.session_state:
            current_tokens = count_tokens(st.session_state.messages)
        else:
            current_tokens = 0
        status = "⚠️ 接近上限" if current_tokens > threshold * 0.8 else "✅ 正常"
        token_display.caption(
            f"📊 上下文 Token 数：**{current_tokens}** / {threshold} {status}"
        )
    # 初始显示
    update_token_display()

    # 配置展示
    with st.expander("⚙️ 当前配置"):
        # 模型名称单独一行，使用 st.markdown 显示完整文本
        st.markdown(f"**模型**: `{st.session_state.config_model_name}`")

        # 其他指标使用 st.metric 并排
        col1, col2 = st.columns(2)
        with col1:
            st.metric("全局阈值", f"{st.session_state.config_score_threshold:.2f}")
            st.metric("最大Token", st.session_state.config_max_tokens)
        with col2:
            st.metric("临时阈值", f"{st.session_state.config_temp_score_threshold:.2f}")
            st.metric("压缩比例", f"{st.session_state.config_target_ratio:.2f}")

        # 关键词用小字显示
        st.caption(f"**触发React关键词**：{st.session_state.config_complex_keywords}")

    st.divider()

    st.markdown("## 📎 会话临时文件")

    # 初始化临时上传器 key
    if "temp_uploader_key" not in st.session_state:
        st.session_state.temp_uploader_key = 0

    uploaded_files = st.file_uploader(
        "上传文件（PDF/TXT/MD）, 最多 3 个",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,     # 支持多文件上传
        key=f"temp_uploader_{st.session_state.temp_uploader_key}",
        label_visibility="visible"
    )

    # 只有存在临时文件时才显示清空按钮
    if "temp_filename" in st.session_state:
        if st.button("🗑️ 清空临时文件", use_container_width=True):
            if "temp_vectorstore" in st.session_state:
                del st.session_state.temp_vectorstore
            if "temp_filename" in st.session_state:
                del st.session_state.temp_filename
            st.session_state.temp_uploader_key += 1
            st.rerun()

    if "temp_filename" in st.session_state:
        filenames = st.session_state.temp_filename.split(", ")
        st.write(f"📄 当前临时文件（共 {len(filenames)} 个）：")
        for name in filenames:
            st.caption(f"  - {name}")
    else:
        st.caption("无临时文件")

    # ---------- 上传处理逻辑 ----------
    if uploaded_files and len(uploaded_files) > 0:
        # 获取当前临时文件列表（如果有）
        existing_filenames = []
        if "temp_filename" in st.session_state and st.session_state.temp_filename:
            existing_filenames = [name.strip() for name in st.session_state.temp_filename.split(",")]

        # 检查总数量是否超过3个
        total_files = len(existing_filenames) + len(uploaded_files)
        if total_files > 3:
            st.toast(
                f"⚠️ 最多只能上传3个文件（当前已有 {len(existing_filenames)} 个，本次上传 {len(uploaded_files)} 个），请减少数量。",
                icon="⚠️")
        else:
            # 检查重复文件名
            new_filenames = [f.name for f in uploaded_files]
            all_filenames = existing_filenames + new_filenames
            if len(all_filenames) != len(set(all_filenames)):
                st.toast("⚠️ 存在重复文件名，请重命名后上传。", icon="⚠️")
            else:
                # 解析新上传的文件
                all_chunks = []
                success_files = []
                failed_files = []
                for file in uploaded_files:
                    try:
                        file_bytes = file.read()
                        chunks = chunk_file_from_bytes(file_bytes, file.name)
                        if chunks:
                            all_chunks.extend(chunks)
                            success_files.append(file.name)
                        else:
                            failed_files.append(file.name)
                    except Exception as e:
                        failed_files.append(file.name)
                        print(f"解析文件 {file.name} 失败: {e}")

                if failed_files:
                    st.toast(f"⚠️ 以下文件解析失败，已跳过：{', '.join(failed_files)}", icon="⚠️")

                if not all_chunks:
                    st.toast("❌ 所有文件都无法解析，请检查文件格式或内容。", icon="❌")
                else:
                    # 如果有成功解析的文件
                    with st.spinner("正在解析并合并文件..."):
                        from src.vectorstore import chunk_file_from_bytes, create_temp_vectorstore

                        # 检查是否已有临时向量库
                        if "temp_vectorstore" in st.session_state and st.session_state.temp_vectorstore is not None:
                            # 追加新 chunks 到现有向量库
                            existing_vs = st.session_state.temp_vectorstore
                            existing_vs.add_documents(all_chunks)
                            # 更新文件名列表
                            new_filename_list = existing_filenames + success_files
                            st.session_state.temp_filename = ", ".join(new_filename_list)
                            st.toast(
                                f"✅ 已追加 {len(success_files)} 个文件，当前共 {len(new_filename_list)} 个文件，总 {len(all_chunks)} 个文本块",
                                icon="📄")
                        else:
                            # 首次上传，创建新的临时向量库
                            temp_vs = create_temp_vectorstore(all_chunks)
                            st.session_state.temp_vectorstore = temp_vs
                            st.session_state.temp_filename = ", ".join(success_files)
                            st.toast(f"✅ 已加载 {len(success_files)} 个文件，共 {len(all_chunks)} 个文本块", icon="📄")

                        # 重置上传器
                        st.session_state.temp_uploader_key += 1
                        st.rerun()

# ========== 定义聊天页面 ==========
def chat_page():
    st.title("智能助手")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": INTRODUCE}
        ]

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    if user_input := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            history = st.session_state.messages[:-1]
            # ---------- 检查是否超限，需要压缩 ----------
            # 计算当前总 token（包括刚添加的用户消息）
            total_tokens = count_tokens(st.session_state.messages)
            if total_tokens > threshold:
                # 显示压缩等待提示（使用 spinner）
                with st.spinner("⏳ 上下文接近上限，正在压缩历史摘要，请稍候..."):
                    # 压缩历史（只压缩历史部分，当前用户消息保留）
                    compressed_history = trim_history(history, max_tokens=threshold, target_ratio=st.session_state.config_target_ratio)
                    # 重建消息列表：压缩后的历史 + 当前用户消息
                    st.session_state.messages = compressed_history + [{"role": "user", "content": user_input}]
                    # 更新 history 为压缩后的历史（供后续生成使用）
                    history = compressed_history
                    st.toast("✅ 压缩完成，正在生成回答...")

            chat_container = st.empty()
            chat_container.markdown("卡卡西四处搜刮中...")

            try:
                stream_gen = None
                # 临时库检索
                temp_vs = st.session_state.get("temp_vectorstore")
                if temp_vs is not None:
                    st.caption("📄 基于临时文件回答")
                    docs_and_scores = temp_vs.similarity_search_with_relevance_scores(user_input, k=4)
                    if docs_and_scores and docs_and_scores[0][1] >= st.session_state.config_temp_score_threshold:
                        docs = [doc for doc, _ in docs_and_scores]
                        stream_gen = rag_chain_with_docs(docs, user_input)
                    else:
                        st.caption("💡 临时文件中未找到相关信息，转为全局检索。")

                # 全局逻辑
                if stream_gen is None:
                    intent = route_query(user_input)
                    need_react = False
                    # 限制 React 只对 rag 和 web 意图生效，chat 不生效
                    if intent in ["rag", "web"]:
                        # 读取配置
                        keywords_str = st.session_state.config_complex_keywords
                        complex_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                        if any(kw in user_input for kw in complex_keywords):
                            need_react = True
                        if intent == "rag":
                            has_match, docs, score = search_with_score(user_input, k=1, score_threshold=0.0)
                            if has_match and score < 0.4:
                                need_react = True

                        # 用于调试
                        # print(f"用户输入: {user_input}")
                        # print(f"intent: {intent}")
                        # print(f"复杂关键词命中: {any(kw in user_input for kw in complex_keywords)}")
                        # print(f"need_react 当前值: {need_react}")
                        # # 打印调用栈，看看谁修改了 need_react
                        # import traceback
                        # traceback.print_stack()

                    if need_react:
                        # 进入 ReAct
                        from src.react_agent import react_agent
                        stream_gen = react_agent(user_input, history)
                    else:
                        # 原有管道
                        if intent == "rag":
                            docs_list = list_documents()
                            if not docs_list:
                                stream_gen = iter(["📭 内部知识库为空，请先在侧边栏上传相关 PDF 文档，然后再次提问。"])
                            else:
                                has_match, docs, score = search_with_score(
                                    user_input,
                                    k=4,
                                    score_threshold=st.session_state.config_score_threshold,
                                )
                                if has_match:
                                    stream_gen = rag_chain_with_docs(docs, user_input)
                                else:
                                    stream_gen = general_chat_stream(user_input, history=history)
                        elif intent == "web":
                            stream_gen = general_chat_stream(user_input, history=history)
                        else:  # chat
                            stream_gen = direct_chat_stream(user_input, history=history)

                # 如果 stream_gen 依然为 None，兜底
                if stream_gen is None:
                    stream_gen = iter(["⚠️ 抱歉，我无法处理这个问题，请重试。"])

                # 流式输出处理
                full_response = ""
                has_content = False
                final_container = None

                # 先清除“思考中”提示（但保留 chat_container 供后续可能的清除）
                chat_container.empty()

                for chunk in stream_gen:
                    has_content = True
                    if chunk.startswith("[THOUGHT]"):
                        content = chunk[9:]
                        st.markdown(render_thought(content), unsafe_allow_html=True)
                    elif chunk.startswith("[ACTION]"):
                        content = chunk[8:]
                        st.markdown(render_action(content), unsafe_allow_html=True)
                    elif chunk.startswith("[OBSERVATION]"):
                        content = chunk[13:]
                        st.markdown(render_observation(content), unsafe_allow_html=True)
                    elif chunk.startswith("[FINAL]"):
                        content = chunk[7:]
                        if final_container is None:
                            final_container = st.empty()
                        full_response += content
                        final_container.markdown(full_response + "▌")
                    else:
                        # 无标签内容（如直接聊天/搜索）：视为最终答案
                        if final_container is None:
                            final_container = st.empty()
                        full_response += chunk
                        final_container.markdown(full_response + "▌")

                # 循环结束后的处理
                if not has_content:
                    # 生成器没有任何输出
                    full_response = "⚠️ 抱歉，我暂时无法生成回答，请稍后重试。"
                    # 如果没有容器，直接用 st.markdown 显示
                    if final_container is None:
                        st.markdown(full_response)
                    else:
                        final_container.markdown(full_response)
                else:
                    # 有内容，但 final_container 可能仍为 None（比如所有 chunk 都是 [THOUGHT] 等，但最后没有 [FINAL]）
                    # 这种情况下，中间步骤已经通过 st.markdown 显示，但最终答案为空
                    if final_container is None:
                        # 如果 full_response 为空，则显示提示；否则显示它
                        if not full_response:
                            st.markdown("⚠️ 回答生成完毕，但未输出有效内容。")
                        else:
                            st.markdown(full_response)
                    else:
                        # 有容器，正常显示最终答案
                        final_container.markdown(full_response)

                # 保存到历史（确保至少有一条消息）
                if not full_response:
                    full_response = "（空白回答）"
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # 回答完成后更新 token 使用数量
                update_token_display()

            except Exception as e:
                st.error(f"请求出错：{str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": "抱歉，发生错误。"})

# ========== 配置导航 ==========
# 创建页面列表
page_chat = st.Page(chat_page, title="聊天", icon="💬")
page_knowledge = st.Page(knowledge_page, title="知识库", icon="📚")
page_settings = st.Page("src/pages/settings.py", title="设置", icon="⚙️")

# 创建导航（顶部显示）
pg = st.navigation([page_chat, page_knowledge, page_settings], position="top")

# 运行当前选中的页面
pg.run()

