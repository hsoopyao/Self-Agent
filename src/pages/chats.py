# ========== 定义聊天页面 ==========
import logging

import streamlit as st

from src.retrieval.task_index import expand_query_with_task_index
from src.agents.react_agent import react_agent
from src.agents.router import route_query
from src.chat.direct_chat import direct_chat_stream
from src.chat.general_chat import general_chat_stream
from src.core.config import INTRODUCE
from src.core.context_manager import count_tokens, trim_history
from src.retrieval.rag_chain import rag_chain_with_docs
from src.retrieval.vectorstore import ensure_vectorstore_loaded, get_documents_by_heading_path
from src.retrieval.vectorstore import (
    list_documents,
    search_with_score,
)
from src.ui.sidebar import update_token_display
from src.ui.ui_components import (
    render_action,
    render_observation,
    render_thought,
)
from src.retrieval.heading_search import (
    search_by_heading_keywords,
    merge_docs,
)

logger = logging.getLogger(__name__)


def chat_page():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": INTRODUCE}
        ]

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("请输入您的问题..."):
        history = list(st.session_state.messages)
        current_user_message = {"role": "user", "content": user_input}
        st.session_state.messages.append(current_user_message)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # ---------- 检查是否超限，需要压缩 ----------
            total_tokens = count_tokens(st.session_state.messages)
            threshold = st.session_state.config_max_tokens
            if total_tokens > threshold:
                with st.spinner("⏳ 上下文接近上限，正在压缩历史摘要，请稍候..."):
                    compressed_history = trim_history(
                        history,
                        max_tokens=threshold,
                        target_ratio=st.session_state.config_target_ratio,
                    )
                    st.session_state.messages = compressed_history + [current_user_message]
                    history = compressed_history
                    st.toast("✅ 压缩完成，正在生成回答...")

            chat_container = st.empty()
            chat_container.markdown("卡卡西四处搜刮中...")

            try:
                stream_gen = None

                def enrich_query_with_history(query: str, history: list) -> str:
                    import re
                    if re.search(r'\.pdf|V\d+\.\d+|\d+\.\d+\.\d+', query, re.IGNORECASE):
                        return query
                    for msg in reversed(history[-6:]):
                        if msg["role"] in ("assistant", "user"):
                            content = msg["content"]
                            match = re.search(
                                r'([\w\-_]+\.pdf|V\d+\.\d+\.\d+\.\d+|[A-Z]_\d+\.\d+\.\d+)',
                                content, re.IGNORECASE
                            )
                            if match:
                                file_ref = match.group(1)
                                if any(kw in query for kw in ["资料", "文件", "文档", "该", "此", "这个"]):
                                    return f"{file_ref} {query}"
                                break
                    return query

                # 全局逻辑
                if stream_gen is None:
                    intent = route_query(user_input, history)
                    current_user_message["intent"] = intent
                    logger.debug(f"{user_input}, intent: {intent}")
                    allow_web = st.session_state.allow_web_switch
                    keywords_str = st.session_state.config_complex_keywords
                    complex_keywords = [kw.strip() for kw in keywords_str.split(",") if kw.strip()]
                    need_react = any(kw in user_input for kw in complex_keywords)

                    if not need_react:
                        if intent == "rag":
                            if not ensure_vectorstore_loaded():
                                stream_gen = iter(["⚠️ 向量库加载失败，无法检索本地知识。"])
                            else:
                                docs_list = list_documents()
                                if not docs_list:
                                    if allow_web:
                                        stream_gen = general_chat_stream(user_input, history=history)
                                    else:
                                        stream_gen = iter(
                                            ["📭 内部知识库为空，请先在侧边栏上传相关 PDF 文档，然后再次提问。"])
                                else:
                                    # ==================================================
                                    # ---------- 1. heading 关键词匹配（优先） ----------
                                    # top_k 从 8 降到 3，减少噪声
                                    # ==================================================
                                    heading_docs = search_by_heading_keywords(user_input, top_k=3)

                                    # ---------- 2. 查询改写 + 向量检索 ----------
                                    enriched_query = enrich_query_with_history(user_input, history)
                                    enriched_query = expand_query_with_task_index(enriched_query)

                                    has_match, docs, score = search_with_score(
                                        enriched_query,
                                        k=5,
                                        score_threshold=st.session_state.config_score_threshold,
                                    )
                                    logger.debug(f"向量检索: 命中={has_match}, docs={len(docs)}, score={score}")

                                    # ==================================================
                                    # ---------- 3. 合并：heading 匹配优先 ----------
                                    # merge_docs(heading_docs, docs) 保持 heading_docs 在前
                                    # 不再做 docs.sort（避免按 page 重排打乱顺序）
                                    # ==================================================
                                    if heading_docs:
                                        docs = merge_docs(heading_docs, docs)
                                        has_match = True
                                        score = max(score, 0.7)
                                        logger.info(f"[RAG] heading 匹配 + 向量补充: {len(docs)} 条")

                                    # ==================================================
                                    # ---------- 4. 章节展开 ----------
                                    # 锚点必须用 heading_docs[0]（heading-search 的 top1）
                                    # 不能用 docs[0]（可能被向量检索的其他结果干扰）
                                    # ==================================================
                                    if has_match:
                                        anchor_doc = heading_docs[0] if heading_docs else docs[0]
                                        heading_path = anchor_doc.metadata.get("heading_path")
                                        logger.info(f"[RAG] 章节展开锚点: {heading_path!r}")

                                        chapter_docs = get_documents_by_heading_path(
                                            heading_path, include_subchapters=True
                                        ) if heading_path else None

                                        use_docs = docs
                                        if chapter_docs and len(chapter_docs) > 1:
                                            use_docs = chapter_docs
                                        elif chapter_docs and len(chapter_docs) == 1 and len(chapter_docs[0].page_content) >= 500:
                                            use_docs = chapter_docs

                                        logger.info(
                                            f"[RAG] 使用 {len(use_docs)} 条 docs"
                                            f"（章展开={len(chapter_docs) if chapter_docs else 0}）"
                                        )

                                        stream_gen = rag_chain_with_docs(use_docs, user_input)
                                    else:
                                        if allow_web:
                                            stream_gen = general_chat_stream(user_input, history=history)
                                        else:
                                            stream_gen = iter(["🔒 内部知识库中没有找到足够相关的信息。"])

                        elif intent == "chat":
                            stream_gen = direct_chat_stream(user_input, history=history)

                        elif intent == "web":
                            if allow_web:
                                logger.debug(f"history: {history}")
                                stream_gen = general_chat_stream(user_input, history=history)
                            else:
                                stream_gen = iter(
                                    ["🔒 未开启联网，无法查询实时信息。请在侧边栏打开「允许联网」开关。"])

                        else:
                            logger.debug("兜底逻辑 ReAct")
                            stream_gen = react_agent(user_input, history, allow_web=allow_web)

                    else:
                        logger.debug("进入 ReAct")
                        stream_gen = react_agent(user_input, history, allow_web=allow_web)

                # 兜底
                if stream_gen is None:
                    stream_gen = iter(["⚠️ 抱歉，我无法处理这个问题，请重试。"])

                # 流式输出处理
                full_response = ""
                has_content = False
                final_container = None

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
                        if final_container is None:
                            final_container = st.empty()
                        full_response += chunk
                        final_container.markdown(full_response + "▌")

                # 循环结束处理
                if not has_content:
                    full_response = "⚠️ 抱歉，我暂时无法生成回答，请稍后重试。"
                    if final_container is None:
                        st.markdown(full_response)
                    else:
                        final_container.markdown(full_response)
                else:
                    if final_container is None:
                        if not full_response:
                            st.markdown("⚠️ 回答生成完毕，但未输出有效内容。")
                        else:
                            st.markdown(full_response)
                    else:
                        final_container.markdown(full_response)

                # 保存历史
                if not full_response:
                    full_response = "（空白回答）"
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                update_token_display()
                st.rerun()

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"聊天页面发生错误: {error_details}")
                st.error("⚠️ 请求处理时出现意外错误，请稍后重试。")