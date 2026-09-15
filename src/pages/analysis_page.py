import difflib
import logging
import re

import pandas as pd
import streamlit as st

from src.retrieval.vectorstore import list_documents, ensure_vectorstore_loaded
from src.agents.requirement_agent import (
    requirement_analysis_stream,
    list_history_versions,
    load_history_version,
)
from src.core.structured_store import list_requirements, update_requirement

logger = logging.getLogger(__name__)


def requirement_analysis_page():
    st.title("📋 需求分析")
    st.caption("选择一份文档，自动生成业务流程 / 逻辑关系 / 需求要点，支持编辑与导出。")

    if not ensure_vectorstore_loaded():
        st.stop()

    docs = list_documents()
    if not docs:
        st.info("📭 知识库为空，请先上传文档。")
        return

    filenames = [d["filename"] for d in docs]

    # ---------- 顶部：文档选择 + 三个按钮 ----------
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        selected = st.selectbox(
            "选择要分析的文档",
            filenames,
            key="ra_doc_select",
            label_visibility="collapsed",
        )
    with col2:
        run = st.button("🚀 开始分析", type="primary", width="stretch")
    with col3:
        rerun = st.button(
            "🔄 重新分析", width="stretch",
            help="忽略缓存，强制重新分析",
        )

    if not selected:
        return

    st.divider()

    # ---------- 执行分析 ----------
    if run or rerun:
        st.subheader("📄 分析报告")
        placeholder = st.empty()
        full_text = ""
        stopped = False

        try:
            with st.spinner("正在分析，请稍候..."):
                for chunk in requirement_analysis_stream(
                    selected, use_cache=not rerun
                ):
                    full_text += chunk

                    # 检测 JSON 分隔符，避免漏出
                    if not stopped and "===REQUIREMENTS_JSON===" in full_text:
                        report_part = full_text.split("===REQUIREMENTS_JSON===", 1)[0].rstrip()
                        placeholder.markdown(_escape_md_special(report_part))
                        stopped = True
                    elif not stopped:
                        placeholder.markdown(full_text)

            # 最终清洗
            display_text = full_text.split("===REQUIREMENTS_JSON===", 1)[0]
            display_text = re.sub(r"```json.*?```", "", display_text, flags=re.DOTALL).strip()
            format_text = _escape_md_special(display_text)
            placeholder.markdown(format_text)
            st.session_state[f"report_{selected}"] = format_text

            if rerun:
                st.toast("✅ 已重新分析", icon="🔄")

        except Exception as e:
            logger.error(f"分析失败: {e}")
            st.error(f"分析失败：{e}")

    # ---------- 历史报告 ----------
    cached_report = st.session_state.get(f"report_{selected}")
    if cached_report and not (run or rerun):
        with st.expander("📄 查看完整报告", expanded=False):
            st.markdown(cached_report)

    # ---------- 需求要点表格 ----------
    items = list_requirements(selected)
    if items:
        st.divider()
        st.subheader("📌 需求要点（可直接编辑）")
        _render_editor(selected, items)
    elif not (run or rerun) and not cached_report:
        st.info("尚未分析此文档，点击「开始分析」生成需求报告。")

    # ---------- 历史版本比对 ----------
    _render_history_section(selected)


def _render_editor(selected: str, items: list):
    df = pd.DataFrame(items)
    edited = st.data_editor(
        df,
        width="stretch",
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
        if st.button("💾 保存修改", width="stretch"):
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
            width="stretch",
        )


def _render_history_section(selected: str):
    """历史版本展示 + 比对"""
    versions = list_history_versions(selected)
    if not versions:
        return

    st.divider()
    with st.expander(f"🕒 历史版本（共 {len(versions)} 份）", expanded=False):
        labels = [f"{v['timestamp']} — {v['count']} 条" for v in versions]

        col1, col2 = st.columns(2)
        with col1:
            idx_new = st.selectbox(
                "新版本", range(len(versions)),
                format_func=lambda i: labels[i],
                key=f"hist_new_{selected}",
            )
        with col2:
            idx_old = st.selectbox(
                "对比基准（旧版本）", range(len(versions)),
                format_func=lambda i: labels[i],
                key=f"hist_old_{selected}",
                index=min(1, len(versions) - 1),
            )

        if st.button("🔍 对比两个版本", key=f"diff_btn_{selected}"):
            v_new = load_history_version(versions[idx_new]["path"])
            v_old = load_history_version(versions[idx_old]["path"])
            _show_diff(v_old.get("items", []), v_new.get("items", []))


def _show_diff(old_items: list, new_items: list):
    old_map = {it["req_id"]: it for it in old_items}
    new_map = {it["req_id"]: it for it in new_items}

    old_ids = set(old_map.keys())
    new_ids = set(new_map.keys())

    added = new_ids - old_ids
    removed = old_ids - new_ids
    common = old_ids & new_ids

    changed = []
    for rid in common:
        if (old_map[rid].get("title") != new_map[rid].get("title")
                or old_map[rid].get("description") != new_map[rid].get("description")):
            changed.append(rid)

    st.markdown(
        f"**新增 {len(added)} 条 · 删除 {len(removed)} 条 · 修改 {len(changed)} 条**"
    )

    if added:
        st.markdown("#### ➕ 新增")
        for rid in sorted(added):
            st.markdown(f"- **{rid}** {new_map[rid]['title']}")

    if removed:
        st.markdown("#### ➖ 删除")
        for rid in sorted(removed):
            st.markdown(f"- **{rid}** {old_map[rid]['title']}")

    if changed:
        st.markdown("#### ✏️ 修改")
        for rid in sorted(changed):
            with st.expander(f"{rid} · {new_map[rid]['title']}"):
                old_desc = old_map[rid].get("description", "")
                new_desc = new_map[rid].get("description", "")
                diff = difflib.unified_diff(
                    old_desc.splitlines(),
                    new_desc.splitlines(),
                    lineterm="", n=1,
                )
                st.code("\n".join(diff), language="diff")

    if not (added or removed or changed):
        st.info("两个版本内容完全一致。")


def _escape_md_special(text: str) -> str:
    """转义可能触发 Markdown 删除线/分隔线的字符"""
    text = text.replace("~~", "\\~\\~")
    text = re.sub(r"(?m)^-{3,}$", lambda m: m.group(0).replace("-", "\\-"), text)
    return text