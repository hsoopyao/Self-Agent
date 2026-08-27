from src.retrieval.vectorstore import get_retriever
from src.core.llm_client import get_llm
from src.core.config import load_prompt

# 加载 Prompt
RAG_SYSTEM = load_prompt("rag_system.txt")
RAG_USER_TEMPLATE = load_prompt("rag_user.txt")

# 初始化 LLM（流式）
llm = get_llm(streaming=True, temperature=0.2)

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