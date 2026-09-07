import streamlit as st
import logging
from src.retrieval.vectorstore import list_documents, add_documents_to_store, delete_document_by_filename, chunk_pdf_from_bytes
from src.retrieval.vectorstore import ensure_vectorstore_loaded
from src.core.memory_manager import get_categories, add_category, delete_category

logger = logging.getLogger(__name__)

def main():
    st.title("📚 知识库")

    # 确保向量库已加载
    if not ensure_vectorstore_loaded():
        st.stop()

    # ---------- 获取所有分类 ----------
    categories = get_categories()  # 从 memory 获取已保存的分类列表

    # ---------- 新建分类（折叠） ----------
    with st.expander("➕ 新建分类", expanded=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            new_cat = st.text_input(
                "输入新分类名称",
                placeholder="例如：项目、工作、学习",
                key="new_category_input",
                label_visibility="collapsed"  # 隐藏标签，避免重复
            )
        with col2:
            if st.button("创建", use_container_width=True):
                if new_cat and new_cat.strip():
                    if add_category(new_cat.strip()):
                        st.success(f"分类 '{new_cat.strip()}' 已创建")
                        st.rerun()
                    else:
                        st.warning("分类已存在或名称为空")
                else:
                    st.warning("请输入分类名称")

    st.divider()

    # ---------- 上传文档 ----------
    st.subheader("上传文档")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader("选择 PDF 文档", type="pdf", key=f"doc_uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        st.caption(f"📎 已选择：{uploaded_file.name}")
        # 下拉选项：若分类列表为空，则默认显示“未分类”（但不保存到分类列表）
        category_options = categories if categories else ["未分类"]
        selected_category = st.selectbox("选择分类", category_options, index=0)

        if st.button("导入文档", use_container_width=True):
            existing_docs = list_documents()
            existing_filenames = [doc["filename"] for doc in existing_docs]
            if uploaded_file.name in existing_filenames:
                st.toast(f"⚠️ 文档 '{uploaded_file.name}' 已存在，请勿重复导入。", icon="⚠️")
            else:
                with st.spinner("正在处理和索引文档..."):
                    file_bytes = uploaded_file.read()
                    chunks = chunk_pdf_from_bytes(file_bytes, uploaded_file.name, selected_category)
                    if add_documents_to_store(chunks):
                        st.toast(f"✅ 文档 '{uploaded_file.name}' 导入成功！", icon="📄")
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.toast("❌ 导入失败，请查看控制台日志。", icon="⚠️")

    st.divider()

    # ---------- 显示文档列表（按分类分组，包含空分类） ----------
    st.markdown("**已导入的文档**")
    docs = list_documents()

    # 构建分类->文档列表映射
    doc_groups = {}
    for doc in docs:
        cat = doc.get("category", "未分类")
        doc_groups.setdefault(cat, []).append(doc["filename"])

    # 合并所有分类（来自 memory 的分类 + 文档中出现的分类）
    all_cats = set(categories) | set(doc_groups.keys())
    if not all_cats:
        st.info("📭 知识库为空，请先创建分类或上传文档。")
    else:
        for cat in sorted(all_cats):
            files = doc_groups.get(cat, [])
            with st.expander(f"📁 {cat} ({len(files)} 个文件)", expanded=False):
                if not files:
                    st.caption("（空文件夹）")
                else:
                    for fname in files:
                        col1, col2 = st.columns([4, 1])
                        col1.write(fname)
                        if col2.button("删除", key=f"del_{fname}"):
                            delete_document_by_filename(fname)
                            st.rerun()

                # 删除分类（使用 popover，紧凑且不占空间）
                # 若分类中有文件，则禁用删除
                disable_delete = len(files) > 0
                help_text = "该分类下还有文件，无法删除" if disable_delete else "删除此分类（不影响已有文档）"
                with st.popover("🗑️", help=help_text, disabled=disable_delete):
                    st.warning(f"确定要删除分类 '{cat}' 吗？")
                    if st.button("确认删除", key=f"confirm_del_{cat}"):
                        if delete_category(cat):
                            st.success(f"分类 '{cat}' 已删除（不影响已有文档）")
                            st.rerun()

if __name__ == "__main__":
    main()