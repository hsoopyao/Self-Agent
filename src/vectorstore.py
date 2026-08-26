import sys
import os
import tempfile
import uuid
from typing import List, Dict
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.load_docs import load_and_chunk_documents

# ---------- 常量配置 ----------
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "D:/Project/bge-small-zh-v1.5/models/BAAI--bge-small-zh-v1.5/snapshots/master"

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
            print(f"成功加载已有向量数据库，共 {vectorstore._collection.count()} 条向量。")
            return vectorstore
        else:
            print("向量库为空，将创建空库。")
            if auto_load:
                print("auto_load 开启，从 data/ 目录加载文档...")
                chunks = load_and_chunk_documents()
                if chunks:
                    vectorstore.add_documents(chunks)
                    print(f"已从 data/ 加载 {len(chunks)} 个文本块。")
                else:
                    print("data/ 目录无文档，保持空库。")
            return vectorstore
    except Exception as e:
        # 目录可能不存在，新建
        print(f"未找到已有向量库，将新建空库。错误: {e}")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        if auto_load:
            print("auto_load 开启，从 data/ 目录加载文档...")
            chunks = load_and_chunk_documents()
            if chunks:
                vectorstore.add_documents(chunks)
                print(f"已从 data/ 加载 {len(chunks)} 个文本块。")
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
def chunk_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
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
        filenames = set()
        for meta in metadata:
            if meta:
                # 优先使用 filename 字段
                filename = meta.get("filename")
                # 若不存在，尝试从 source 提取（兼容旧数据）
                if not filename:
                    source = meta.get("source")
                    if source:
                        filename = os.path.basename(source)
                if filename:
                    filenames.add(filename)
        return [{"id": fname, "filename": fname} for fname in filenames]
    except Exception as e:
        print(f"获取文档列表失败: {e}")
        return []

def delete_document_by_filename(filename: str) -> bool:
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection
        collection.delete(where={"filename": filename})
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

# ---------- 临时向量库 ----------
def create_temp_vectorstore(chunks: List[Document]):
    # 增加空列表检查，避免Chroma报错
    if not chunks:
        raise ValueError("无法创建空的临时向量库，请检查文档内容。")
    embeddings = get_embeddings()
    collection_name = f"temp_{uuid.uuid4().hex[:8]}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None,
        collection_name=collection_name
    )
    return vectorstore