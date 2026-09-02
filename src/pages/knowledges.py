import streamlit as st
import logging

from src.retrieval.vectorstore import list_documents, add_documents_to_store, delete_document_by_filename, chunk_pdf_from_bytes

logger = logging.getLogger(__name__)

def main():
    st.title("📚 知识库")

    # 定义预置分类
    CATEGORIES = ["项目", "工作", "学习", "内部政策资料", "未分类"]

    # 初始化上传器计数器
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader("上传PDF文档", type="pdf", key=f"doc_uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        st.caption(f"📎 已选择：{uploaded_file.name}")
        category = st.selectbox("文档分类", CATEGORIES, index=len(CATEGORIES) - 1)
        if st.button("导入文档", use_container_width=True):
            existing_docs = list_documents()
            existing_filenames = [doc["filename"] for doc in existing_docs]
            if uploaded_file.name in existing_filenames:
                st.toast(f"⚠️ 文档 '{uploaded_file.name}' 已存在，请勿重复导入。", icon="⚠️")
            else:
                with st.spinner("正在处理和索引文档..."):
                    file_bytes = uploaded_file.read()
                    chunks = chunk_pdf_from_bytes(file_bytes, uploaded_file.name, category)
                    if add_documents_to_store(chunks):
                        st.toast(f"✅ 文档 '{uploaded_file.name}' 导入成功！", icon="📄")
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.toast("❌ 导入失败，请查看控制台日志。", icon="⚠️")

    st.markdown("**已导入的文档**")
    docs = list_documents()
    if not docs:
        st.info("📭 知识库为空，请上传文档。")
    else:
        # 按分类分组
        grouped = {}
        for doc in docs:
            cat = doc.get("category", "未分类")
            grouped.setdefault(cat, []).append(doc["filename"])

        # 按固定分类顺序显示（仅显示非空分类）
        for cat in CATEGORIES:
            if cat in grouped and grouped[cat]:
                # 使用 st.expander 创建可折叠节点
                with st.expander(f"📁 {cat} ({len(grouped[cat])} 个文件)", expanded=False):
                    for fname in grouped[cat]:
                        col1, col2 = st.columns([4, 1])
                        col1.write(fname)
                        if col2.button("删除", key=f"del_{fname}"):
                            delete_document_by_filename(fname)
                            st.rerun()

if __name__ == "__main__":
    main()