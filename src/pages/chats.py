# ========== 定义聊天页面 ==========
import logging

import streamlit as st

from src.core.context_manager import count_tokens, trim_history
from src.core.config import INTRODUCE
from src.chat.direct_chat import direct_chat_stream
from src.chat.general_chat import general_chat_stream
from src.retrieval.rag_chain import rag_chain_with_docs
from src.agents.router import route_query
from src.agents.react_agent import react_agent
from src.ui.sidebar import update_token_display
from src.ui.ui_components import (
    render_action,
    render_observation,
    render_thought,
)
from src.retrieval.vectorstore import (
    list_documents,
    search_with_score,
)

logger = logging.getLogger(__name__)

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
        # 先保存进入本轮前的历史，再单独追加当前问题。
        # 这样生成阶段无需依赖 messages[:-1]，也不会重复传入当前问题。
        history = list(st.session_state.messages)
        current_user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(current_user_message)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # ---------- 检查是否超限，需要压缩 ----------
            # 计算当前总 token（包括刚添加的用户消息）
            total_tokens = count_tokens(st.session_state.messages)
            threshold = st.session_state.config_max_tokens
            if total_tokens > threshold:
                # 显示压缩等待提示（使用 spinner）
                with st.spinner("⏳ 上下文接近上限，正在压缩历史摘要，请稍候..."):
                    # 压缩历史（只压缩历史部分，当前用户消息保留）
                    compressed_history = trim_history(history, max_tokens=threshold, target_ratio=st.session_state.config_target_ratio)
                    # 重建消息列表：压缩后的历史 + 当前用户消息
                    st.session_state.messages = compressed_history + [current_user_message]
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
                    logger.debug(f"{user_input}, intent: {intent}")
                    # 读取关键词配置
                    keywords_str = st.session_state.config_complex_keywords
                    complex_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                    # 搜索内部文件但无对比
                    if intent == "rag" and not any(kw in user_input for kw in complex_keywords):
                        docs_list = list_documents()
                        if not docs_list:
                            if allow_web:
                                stream_gen = general_chat_stream(user_input, history=history)
                            else:
                                stream_gen = iter(["📭 内部知识库为空，请先在侧边栏上传相关 PDF 文档，然后再次提问。"])
                        else:
                            has_match, docs, score = search_with_score(
                                user_input,
                                k=4,
                                score_threshold=st.session_state.config_score_threshold,
                            )
                            logger.debug(f'{has_match}, docs: {len(docs)}, score: {score}')
                            if has_match:
                                stream_gen = rag_chain_with_docs(docs, user_input)
                            else:
                                if allow_web:
                                    logger.debug("no match but allow web...")
                                    stream_gen = general_chat_stream(user_input, history=history)
                                else:
                                    stream_gen = iter([
                                        "🔒 内部知识库中没有找到足够相关的信息，本次未自动发送到外部网络。"
                                        "如需继续，请在问题中明确写明“联网搜索”。"
                                    ])
                    elif intent == "chat":
                        from src.core.memory_manager import get_all_memories
                        memories = get_all_memories()
                        memory_context = ""
                        if memories:
                            memory_context = "；".join([f"{k}:{v}" for k, v in memories.items()])
                        stream_gen = direct_chat_stream(user_input, history, memory_context=memory_context)
                    else:
                        logger.debug("进入 ReAct")
                        stream_gen = react_agent(
                            user_input,
                            history,
                            allow_web=allow_web,
                        )

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

                # 将本轮动态流式占位符转换为稳定的历史消息。
                # 否则下次提交问题时，Streamlit 会在处理期间保留上一轮的灰色旧占位符，
                # 与上方重新渲染的历史回答形成视觉重复。
                st.rerun()


            except Exception as e:
                # 记录详细错误日志
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"聊天页面发生错误: {error_details}")
                # 用户友好提示
                st.error("⚠️ 处理请求时出现意外错误，请稍后重试。")