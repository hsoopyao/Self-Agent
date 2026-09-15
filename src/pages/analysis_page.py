import logging
import re

import pandas as pd
import streamlit as st

from src.retrieval.vectorstore import list_documents, ensure_vectorstore_loaded
from src.agents.requirement_agent import requirement_analysis_stream
from src.core.structured_store import list_requirements, update_requirement

logger = logging.getLogger(__name__)


def requirement_analysis_page():
    st.title("📋 需求分析")
    st.caption("选择一份文档，自动生成业务流程 / 逻辑关系 / 需求要点，支持编辑与导出。")

    if not ensure_vectorstore_loaded():
        st.stop()

    # ---------- 文档选择 ----------
    docs = list_documents()
    if not docs:
        st.info("📭 知识库为空，请先上传文档。")
        return

    filenames = [d["filename"] for d in docs]

    col1, col2 = st.columns([4, 1])
    with col1:
        selected = st.selectbox(
            "选择要分析的文档",
            filenames,
            key="ra_doc_select",
            label_visibility="collapsed",
        )
    with col2:
        run = st.button("🚀 开始分析", type="primary", use_container_width=True)

    if not selected:
        return

    st.divider()

    # ---------- 执行分析 ----------
    if run:
        st.subheader("📄 分析报告")
        placeholder = st.empty()
        full_text = ""
        try:
            with st.spinner("正在分析，请稍候..."):
                for chunk in requirement_analysis_stream(selected):
                    full_text += chunk
                    placeholder.markdown(full_text)
            # 清除报告尾部的json(用于入库)
            display_text = re.sub(r"```json.*?```", "", full_text, flags=re.DOTALL).strip()
            format_text = _escape_md_special(display_text)
            placeholder.markdown(format_text)
            # 缓存到 session_state，切 tab 回来还在
            st.session_state[f"report_{selected}"] = format_text
        except Exception as e:
            logger.error(f"分析失败: {e}")
            st.error(f"分析失败：{e}")

    # ---------- 历史报告（切换文档或切 tab 后仍可见） ----------
    cached_report = st.session_state.get(f"report_{selected}")
    if cached_report and not run:
        with st.expander("📄 查看完整报告", expanded=False):
            st.markdown(cached_report)

    # ---------- 需求要点表格 ----------
    items = list_requirements(selected)
    if items:
        st.divider()
        st.subheader("📌 需求要点（可直接编辑）")

        df = pd.DataFrame(items)
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key=f"ra_editor_{selected}",
            column_config={
                "req_id": st.column_config.TextColumn("编号", width="small"),
                "title": st.column_config.TextColumn("标题", width="medium"),
                "description": st.column_config.TextColumn("描述", width="large"),
                "acceptance": st.column_config.TextColumn("验收标准", width="large"),
                "source": st.column_config.TextColumn("来源", width="medium"),
                "priority": st.column_config.TextColumn("优先级", width="small"),
            },
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存修改", use_container_width=True):
                for _, row in edited.iterrows():
                    update_requirement(selected, row["req_id"], row.to_dict())
                st.success(f"已保存 {len(edited)} 条")
        with col2:
            csv = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 导出 CSV",
                csv,
                file_name=f"{selected}_需求清单.csv",
                mime="text/csv",
                use_container_width=True,
            )
    elif not run and not cached_report:
        st.info("尚未分析此文档，点击「开始分析」生成需求报告。")

def _escape_md_special(text: str) -> str:
    """转义可能触发 Markdown 删除线/分隔线的字符"""
    # ~~ 删除线
    text = text.replace("~~", "\\~\\~")
    # 连续 --- 单独成行会变分隔线
    text = re.sub(r"(?m)^-{3,}$", lambda m: m.group(0).replace("-", "\\-"), text)
    return text