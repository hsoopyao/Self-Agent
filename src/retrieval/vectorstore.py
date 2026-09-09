import os
import tempfile
import logging
from typing import List, Dict
import streamlit as st

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pdfmuse import PdfmuseLoader

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

# 获取 embedding 模型单例
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings

# 全局单例：获取向量库实例（仅初始化一次）
def get_vectorstore():
    global _vectorstore, _initialized
    if _vectorstore is None:
        if not _initialized:
            _vectorstore = create_vectorstore()
            _initialized = True
        else:
            _vectorstore = create_vectorstore(auto_load=False)
    return _vectorstore

# ---------- 格式化文档函数 ----------
def _clean_metadata_for_chroma(doc: Document) -> Document:
    """
    清理 Document 的 metadata，使其符合 Chroma 的值类型要求：
    - 只保留 str, int, float, bool, list, None 类型的值
    - 对于 list，确保非空且元素都是标量；否则删除该键
    - 删除 dict 类型的值（如 bbox）
    - 其他类型转为字符串
    """
    cleaned_meta = {}
    for key, value in doc.metadata.items():
        if isinstance(value, dict):
            continue
        elif isinstance(value, list):
            # 检查列表是否非空且所有元素都是标量（str/int/float/bool）
            if value and all(isinstance(v, (str, int, float, bool)) for v in value):
                cleaned_meta[key] = value
            else:
                # 空列表或包含不可序列化元素，跳过该键
                continue
        elif isinstance(value, (str, int, float, bool, type(None))):
            cleaned_meta[key] = value
        else:
            cleaned_meta[key] = str(value)
    heading_path = doc.metadata.get("heading_path")
    if isinstance(heading_path, list) and heading_path:
        cleaned_meta["heading_path_str"] = " > ".join(heading_path)
    return Document(page_content=doc.page_content, metadata=cleaned_meta)

def _flatten_docs(docs):
    """递归展平嵌套列表，并将所有元素转换为 Document（如果还不是）"""
    flat = []
    for item in docs:
        if isinstance(item, list):
            flat.extend(_flatten_docs(item))
        elif isinstance(item, Document):
            flat.append(item)
        else:
            flat.append(_to_document(item))
    return flat

def _to_document(item):
    """将非 Document 对象转为 Document（兼容旧代码）"""
    if isinstance(item, Document):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        content, meta = item
        if isinstance(meta, dict):
            return Document(page_content=content, metadata=meta)
    return Document(page_content=str(item))

# ---------- 向量库加载函数 ----------
def create_vectorstore(auto_load: bool = False):
    embeddings = get_embeddings()
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        if vectorstore._collection.count() > 0:
            logger.info(f"成功加载已有向量数据库，共 {vectorstore._collection.count()} 条向量。")
            return vectorstore
        else:
            logger.info("向量库为空，将创建空库。")
            if auto_load:
                chunks = load_and_chunk_documents()
                if chunks:
                    flat = _flatten_docs(chunks)
                    if flat:
                        cleaned = [_clean_metadata_for_chroma(doc) for doc in flat]
                        vectorstore.add_documents(cleaned)
                        logger.info(f"已从 data/ 加载 {len(cleaned)} 个文本块。")
            return vectorstore
    except Exception as e:
        logger.error(f"未找到已有向量库，将新建空库。错误: {e}")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        if auto_load:
            chunks = load_and_chunk_documents()
            if chunks:
                flat = _flatten_docs(chunks)
                if flat:
                    cleaned = [_clean_metadata_for_chroma(doc) for doc in flat]
                    vectorstore.add_documents(cleaned)
        return vectorstore

def ensure_vectorstore_loaded() -> bool:
    """
    确保向量库已加载，并显示加载状态（适用于 Streamlit）。
    """
    if st.session_state.get("vectorstore_loaded", False):
        return True

    if st.session_state.get("vectorstore_error"):
        st.error(f"⚠️ 向量库加载失败：{st.session_state.vectorstore_error}")
        return False

    try:
        with st.spinner("⏳ 正在加载向量库和 Embedding 模型，请稍候..."):
            _ = get_vectorstore()
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

def get_documents_by_heading_path(heading_path: List[str], include_subchapters: bool = True) -> List[Document]:
    """
    根据 heading_path 列表，获取该章节下所有文档块（包括子章节），按页码排序。
    需要 metadata 中含有 'heading_path_str' 字段。
    """
    vectorstore = get_vectorstore()
    # 获取全部数据（若数据量巨大，可考虑分页，但通常几千块内没问题）
    all_data = vectorstore.get(include=["documents", "metadatas"])
    target_prefix = " > ".join(heading_path)
    matched = []
    for content, meta in zip(all_data['documents'], all_data['metadatas']):
        path_str = meta.get("heading_path_str", "")
        if include_subchapters:
            if path_str == target_prefix or path_str.startswith(target_prefix + " > "):
                matched.append(Document(page_content=content, metadata=meta))
        else:
            if path_str == target_prefix:
                matched.append(Document(page_content=content, metadata=meta))
    matched.sort(key=lambda d: d.metadata.get("page", 0))
    return matched

# ---------- 知识库导入函数 ----------
def chunk_pdf_from_bytes(file_bytes: bytes, filename: str, category: str = "未分类") -> List[Document]:
    """
    从字节流中解析 PDF，直接返回 PdfmuseLoader 的原始元素块，不做二次切分。
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PdfmuseLoader(tmp_path, mode="elements")
        docs = loader.load()

        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["user_category"] = category
        return docs  # 直接返回，不做切分

    finally:
        os.unlink(tmp_path)

def add_documents_to_store(docs: List[Document]) -> bool:
    """
    将文档列表添加到向量库，自动清理不支持的元数据类型。
    """
    try:
        flat = _flatten_docs(docs)
        if not flat:
            logger.warning("传入的文档列表为空或没有 Document 对象")
            return False
        cleaned = [_clean_metadata_for_chroma(doc) for doc in flat]
        vectorstore = get_vectorstore()
        vectorstore.add_documents(cleaned)
        global _vectorstore
        _vectorstore = None
        return True
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        return False

# ---------- 知识库文档分类函数 ----------
def get_collection():
    vectorstore = get_vectorstore()
    return vectorstore._collection


def list_documents() -> List[Dict[str, str]]:
    """
    列出向量库中所有文档的文件名和用户分类。
    """
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
                cat = meta.get("user_category", meta.get("category", "未分类"))
                if filename not in doc_map:
                    doc_map[str(filename)] = cat
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
        global _vectorstore
        _vectorstore = None
        return True
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        return False