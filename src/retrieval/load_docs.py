"""
负责加载指定目录下的所有文档（PDF），并将其切分成适合检索的文本块。
"""
import os
import logging
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_and_chunk_documents(data_dir: str = "data") -> List[Document]:
    all_docs = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            logger.info(f"正在加载：{file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # 为每页文档添加文件名元数据
            for doc in docs:
                doc.metadata["filename"] = file
            all_docs.extend(docs)

    if not all_docs:
        logger.warning("警告：未找到任何 PDF 文件，请检查 data 目录。")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"共生成 {len(chunks)} 个文本块")
    return chunks