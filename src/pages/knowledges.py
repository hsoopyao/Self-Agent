import streamlit as st
import logging
from src.retrieval.vectorstore import list_documents, add_documents_to_store, delete_document_by_filename, chunk_pdf_from_bytes
from src.retrieval.vectorstore import ensure_vectorstore_loaded

logger = logging.getLogger(__name__)

def main():
    st.title("📚 知识库")

    # 确保向量库已加载
    if not ensure_vectorstore_loaded():
        st.stop()

    # ---------- 上传文档 ----------
    st.subheader("上传文档")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # 获取已有文档的分类列表（动态）
    existing_docs = list_documents()
    existing_categories = sorted({doc.get("category", "未分类") for doc in existing_docs})
    # 确保至少有“未分类”选项
    if not existing_categories:
        existing_categories = ["未分类"]

    # 添加“新建分类...”选项
    category_options = existing_categories + ["➕ 新建分类..."]

    uploaded_file = st.file_uploader("选择 PDF 文档", type="pdf", key=f"doc_uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        st.caption(f"📎 已选择：{uploaded_file.name}")

        # 分类选择
        selected = st.selectbox("选择分类", category_options, index=0)

        # 如果选择“新建分类...”，显示输入框
        if selected == "➕ 新建分类...":
            new_category = st.text_input(
                "输入新分类名称",
                placeholder="例如：项目、工作、学习",
                key="new_category_input"
            )
            final_category = new_category.strip() if new_category.strip() else "未分类"
        else:
            final_category = selected

        if st.button("导入文档", use_container_width=True):
            existing_filenames = [doc["filename"] for doc in existing_docs]
            if uploaded_file.name in existing_filenames:
                st.toast(f"⚠️ 文档 '{uploaded_file.name}' 已存在，请勿重复导入。", icon="⚠️")
            else:
                with st.spinner("正在处理和索引文档..."):
                    file_bytes = uploaded_file.read()
                    chunks = chunk_pdf_from_bytes(file_bytes, uploaded_file.name, final_category)
                    if add_documents_to_store(chunks):
                        st.toast(f"✅ 文档 '{uploaded_file.name}' 导入成功！", icon="📄")
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.toast("❌ 导入失败，请查看控制台日志。", icon="⚠️")

    st.divider()

    # ---------- 显示文档列表（按分类分组） ----------
    st.markdown("**已导入的文档**")
    docs = list_documents()
    if not docs:
        st.info("📭 知识库为空，请上传文档。")
    else:
        grouped = {}
        for doc in docs:
            cat = doc.get("category", "未分类")
            grouped.setdefault(cat, []).append(doc["filename"])

        for cat in sorted(grouped.keys()):
            files = grouped[cat]
            with st.expander(f"📁 {cat} ({len(files)} 个文件)", expanded=False):
                for fname in files:
                    col1, col2 = st.columns([4, 1])
                    col1.write(fname)
                    if col2.button("删除", key=f"del_{fname}"):
                        delete_document_by_filename(fname)
                        st.rerun()

if __name__ == "__main__":
    main()