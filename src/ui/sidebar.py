import streamlit as st

from src.core.config import INTRODUCE
from src.core.context_manager import count_tokens
from src.retrieval.vectorstore import chunk_file_from_bytes, create_temp_vectorstore, delete_temp_file_by_filename

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
        # 会话管理
        st.markdown("### 🧹 会话管理")
        if st.button("🗑️ 清空上下文窗口", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": INTRODUCE}]
            st.rerun()

        st.session_state.token_display = st.empty()
        update_token_display()

        # 配置展示
        with st.expander("⚙️ 当前配置"):
            st.markdown(f"**模型**: `{st.session_state.config_model_name}`")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("全局阈值", f"{st.session_state.config_score_threshold:.2f}")
                st.metric("最大Token", st.session_state.config_max_tokens)
            with col2:
                st.metric("临时阈值", f"{st.session_state.config_temp_score_threshold:.2f}")
                st.metric("压缩比例", f"{st.session_state.config_target_ratio:.2f}")
            st.metric("ReAct最大步数", st.session_state.config_react_max_steps)
            st.caption(f"**触发React关键词**：{st.session_state.config_complex_keywords}")

        st.divider()

        st.markdown("## 📎 会话临时文件")

        # 初始化临时上传器 key
        if "temp_uploader_key" not in st.session_state:
            st.session_state.temp_uploader_key = 0

        uploaded_files = st.file_uploader(
            "上传文件（PDF/TXT/MD）, 最多 3 个",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,  # 支持多文件上传
            key=f"temp_uploader_{st.session_state.temp_uploader_key}",
            label_visibility="visible"
        )

        # 只有存在临时文件时才显示清空按钮
        if "temp_filename" in st.session_state:
            filenames = st.session_state.temp_filename.split(", ")
            st.write(f"📄 当前临时文件（共 {len(filenames)} 个）：")
            for idx, name in enumerate(filenames):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(f"  - {name}")
                with col2:
                    # 每个文件的删除按钮，使用唯一的 key
                    if st.button("✕", key=f"del_temp_{idx}_{name}"):
                        if delete_temp_file_by_filename(name):
                            st.toast(f"已删除临时文件：{name}", icon="🗑️")
                            st.rerun()  # 刷新
                        else:
                            st.toast(f"删除失败：{name}", icon="❌")
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
                                st.toast(f"✅ 已加载 {len(success_files)} 个文件，共 {len(all_chunks)} 个文本块",
                                         icon="📄")

                            # 重置上传器
                            st.session_state.temp_uploader_key += 1
                            st.rerun()

        # 最后将 update_token_display 暴露给外部（以便在 chat_page 完成后调用）
        return update_token_display
