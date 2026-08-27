# pages/knowledges.py
import streamlit as st

from src.retrieval.vectorstore import list_documents, add_documents_to_store, delete_document_by_filename, chunk_pdf_from_bytes

def main():
    st.title("📚 知识库")

    # 初始化上传器计数器
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader("上传PDF文档", type="pdf", key=f"doc_uploader_{st.session_state.uploader_key}")
    if uploaded_file is not None:
        st.caption(f"📎 已选择：{uploaded_file.name}")
        if st.button("导入文档", use_container_width=True):
            existing_docs = list_documents()
            existing_filenames = [doc["filename"] for doc in existing_docs]
            if uploaded_file.name in existing_filenames:
                st.toast(f"⚠️ 文档 '{uploaded_file.name}' 已存在，请勿重复导入。", icon="⚠️")
            else:
                with st.spinner("正在处理和索引文档..."):
                    file_bytes = uploaded_file.read()
                    chunks = chunk_pdf_from_bytes(file_bytes, uploaded_file.name)
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
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {doc['filename']}")
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}"):
                    if delete_document_by_filename(doc['filename']):
                        st.toast(f"✅ 已删除 {doc['filename']}", icon="🗑️")
                        st.rerun()
                    else:
                        st.toast(f"❌ 删除失败：{doc['filename']}", icon="⚠️")

if __name__ == "__main__":
    main()