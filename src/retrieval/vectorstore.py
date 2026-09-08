import os
import streamlit as st
import tempfile
import logging
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.retrieval.load_docs import load_and_chunk_documents

logger = logging.getLogger(__name__)

# ---------- 常量配置 ----------
PERSIST_DIR = "./db/chroma_db"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_PATH",
    "BAAI/bge-small-zh-v1.5"
)

# ---------- 全局单例 ----------
_vectorstore = None
_embeddings = None
_initialized = False

def get_embeddings():
    """获取 embedding 模型单例"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

def create_vectorstore(auto_load: bool = True):
    """
    创建或加载向量数据库。
    auto_load: 如果为 True，则在向量库为空时从 data/ 目录加载文档。
    默认改为 True，启动时若为空则自动加载 data 目录中的 PDF。
    """
    embeddings = get_embeddings()

    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        if vectorstore._collection.count() > 0:
            logger.info(f"成功加载已有向量数据库，共 {vectorstore._collection.count()} 条向量。")
            return vectorstore
        else:
            logger.info("向量库为空，将创建空库。")
            if auto_load:
                logger.info("auto_load 开启，从 data/ 目录加载文档...")
                chunks = load_and_chunk_documents()
                if chunks:
                    vectorstore.add_documents(chunks)
                    logger.info(f"已从 data/ 加载 {len(chunks)} 个文本块。")
                else:
                    logger.info("data/ 目录无文档，保持空库。")
            return vectorstore
    except Exception as e:
        # 目录可能不存在，新建
        logger.error(f"未找到已有向量库，将新建空库。错误: {e}")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        if auto_load:
            logger.info("auto_load 开启，从 data/ 目录加载文档...")
            chunks = load_and_chunk_documents()
            if chunks:
                vectorstore.add_documents(chunks)
                logger.info(f"已从 data/ 加载 {len(chunks)} 个文本块。")
        return vectorstore

def get_vectorstore():
    """全局单例：获取向量库实例（仅初始化一次）"""
    global _vectorstore, _initialized
    if _vectorstore is None:
        if not _initialized:
            _vectorstore = create_vectorstore()  # 启动时自动加载
            _initialized = True
        else:
            _vectorstore = create_vectorstore(auto_load=False)
    return _vectorstore

def ensure_vectorstore_loaded() -> bool:
    """
    确保向量库已加载，并显示加载状态。
    返回 True 表示加载成功，False 表示失败。
    """
    # 如果已经加载成功，直接返回
    if st.session_state.get("vectorstore_loaded", False):
        return True

    # 如果之前加载失败，直接返回 False，不重复尝试（可添加重试逻辑）
    if st.session_state.get("vectorstore_error"):
        st.error(f"⚠️ 向量库加载失败：{st.session_state.vectorstore_error}")
        return False

    # 开始加载
    try:
        with st.spinner("⏳ 正在加载向量库和 Embedding 模型，请稍候..."):
            # 实际加载（可能会耗时）
            _ = get_vectorstore()  # 触发加载
        st.session_state.vectorstore_loaded = True
        st.session_state.vectorstore_error = None
        st.toast("✅ 向量库加载成功", icon="✅")
        return True
    except Exception as e:
        st.session_state.vectorstore_error = str(e)
        st.error(f"❌ 向量库加载失败：{e}")
        return False

# ---------- 检索函数 ----------
def get_retriever(vectorstore=None, k: int = 2):
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})

def search_with_score(query: str, k: int = 2, score_threshold: float = 0.5):
    vectorstore = get_vectorstore()
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    if not docs_and_scores:
        return False, [], 0.0
    top_score = docs_and_scores[0][1]
    if top_score < score_threshold:
        return False, [], top_score
    docs = [doc for doc, _ in docs_and_scores]
    return True, docs, top_score

# ---------- 知识库导入函数 ----------
def chunk_pdf_from_bytes(file_bytes: bytes, filename: str, category: str = "未分类") -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["filename"] = filename
        chunk.metadata["category"] = category
    return chunks

# ---------- 临时会话文件导入函数 ----------
def chunk_file_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    从字节流加载文件（支持 PDF、TXT、MD），切分成文本块，并添加 filename 元数据。
    """
    ext = os.path.splitext(filename)[1].lower()
    docs = []

    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            os.unlink(tmp_path)
    elif ext in [".txt", ".md"]:
        # 文本文件直接读取
        text = file_bytes.decode("utf-8", errors="ignore")
        # 创建一个虚拟 Document
        from langchain_core.documents import Document
        docs = [Document(page_content=text, metadata={"source": filename})]
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(docs)

    # 为每个块添加 filename 元数据
    for chunk in chunks:
        chunk.metadata["filename"] = filename

    return chunks

def get_collection():
    vectorstore = get_vectorstore()
    return vectorstore._collection

def list_documents() -> List[Dict[str, str]]:
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        result = collection.get()
        metadata = result.get("metadatas", [])
        doc_map = {}
        for meta in metadata:
            if not meta:
                continue
            filename = meta.get("filename")
            if not filename:
                source = meta.get("source")
                if source:
                    filename = os.path.basename(source)
            if filename:
                category = meta.get("category", "未分类")
                # 如果同一个文件有多个chunk，保留首次遇到的分类（假设同文件分类一致）
                if filename not in doc_map:
                    doc_map[str(filename)] = category
        # 转换为列表
        return [{"filename": fname, "category": cat} for fname, cat in doc_map.items()]
    except Exception as e:
        logger.error(f"获取文档列表失败: {e}")
        return []

def delete_document_by_filename(filename: str) -> bool:
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        collection.delete(where={
            "$or": [
                {"source": filename},
                {"filename": filename}
            ]
        })
        # 清空全局单例，强制下次重新加载
        global _vectorstore
        _vectorstore = None
        return True
    except Exception as e:
        print(f"删除文档失败: {e}")
        return False

def add_documents_to_store(docs: List[Document]) -> bool:
    try:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(docs)
        global _vectorstore
        _vectorstore = None  # 强制重新加载，以便后续检索包含新数据
        return True
    except Exception as e:
        print(f"添加文档失败: {e}")
        return False