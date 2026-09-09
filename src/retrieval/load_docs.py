import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_pdfmuse import PdfmuseLoader

logger = logging.getLogger(__name__)


def load_and_chunk_documents(data_dir: str = "data") -> List[Document]:
    """
    加载 data_dir 下所有 PDF，按元素（标题/段落/表格）解析，
    每个元素作为一个独立的 Document，不做二次切分。
    """
    all_elements = []

    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            logger.info(f"正在加载：{file_path}")

            loader = PdfmuseLoader(file_path, mode="elements")
            docs = loader.load()

            for doc in docs:
                doc.metadata["filename"] = file

            all_elements.extend(docs)

    if not all_elements:
        logger.warning("警告：未找到任何 PDF 文件，请检查 data 目录。")
        return []

    logger.info(f"共生成 {len(all_elements)} 个语义元素块（标题/段落/表格）")
    return all_elements
