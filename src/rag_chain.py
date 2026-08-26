import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("BIGMODEL_API_KEY")
LLM_MODEL = st.session_state.get("config_model_name", os.getenv("MODEL_NAME", "glm-4-flash"))
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

if not API_KEY:
    raise ValueError("❌ 未找到 BIGMODEL_API_KEY，请检查 .env 文件。")
os.environ["OPENAI_API_KEY"] = API_KEY

# 加载 Prompt
def load_prompt(filename):
    with open(f"prompts/{filename}", "r", encoding="utf-8") as f:
        return f.read().strip()
    return None
RAG_SYSTEM = load_prompt("rag_system.txt")
RAG_USER_TEMPLATE = load_prompt("rag_user.txt")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.vectorstore import get_retriever

# 初始化 LLM（流式）
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.2,
    streaming=True, # 开启流式
)

def rag_chain_stream(input_dict: dict):
    """
    流式生成器：检索内部文档，流式生成回答。
    """
    retriever = get_retriever(k=4)

    question = input_dict["input"]
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [
        ("system", RAG_SYSTEM),
        ("human", f"参考资料：\n{context}\n\n用户问题：{question}")
    ]
    for chunk in llm.stream(messages):
        yield chunk.content

def rag_chain_with_docs(docs, question: str):
    """
    基于已有的文档列表生成流式回答（不再次检索）
    """
    context = "\n\n".join([doc.page_content for doc in docs])
    messages = [
        ("system", RAG_SYSTEM),
        ("human", RAG_USER_TEMPLATE.format(context=context, question=question))
    ]
    for chunk in llm.stream(messages):
        yield chunk.content