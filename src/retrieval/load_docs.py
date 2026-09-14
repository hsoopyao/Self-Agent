import os
import logging
from typing import List
from langchain_core.documents import Document

from src.retrieval.parse_pdf import parse_pdf_to_chunks

logger = logging.getLogger(__name__)

def load_and_chunk_documents(data_dir: str = "data") -> List[Document]:
    all_chunks = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            logger.info(f"正在解析：{file_path}")
            chunks = parse_pdf_to_chunks(file_path, file)
            all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("data 目录下未找到 PDF")
        return []

    logger.info(f"共生成 {len(all_chunks)} 个语义 chunk")
    return all_chunks