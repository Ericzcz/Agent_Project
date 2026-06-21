import os
import subprocess
from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())


def get_project_root() -> Path:
    """
    获取当前 Git 仓库根目录。
    例如：/workspaces/123
    """
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True
        ).strip()
    )


def load_documents() -> List[Document]:
    """
    从 llm-universe/data_base/knowledge_db 加载 PDF 和 Markdown 文件。
    """
    project_root = get_project_root()
    folder_path = project_root / "llm-universe" / "data_base" / "knowledge_db"

    files_path = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_path.append(file_path)

    loaders = []

    for file_path in files_path:
        suffix = file_path.split(".")[-1].lower()

        if suffix == "pdf":
            loaders.append(PyMuPDFLoader(file_path))

        elif suffix == "md":
            loaders.append(UnstructuredMarkdownLoader(file_path))

    texts = []

    for loader in loaders:
        texts.extend(loader.load())

    return texts


def split_documents(texts: List[Document]) -> List[Document]:
    """
    把原始文档切成 chunks。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return text_splitter.split_documents(texts)


def load_vectorstore() -> Chroma:
    """
    从本地 Chroma 持久化目录加载向量库。
    注意：这里不会重新向量化，也不会 add_documents。
    """
    project_root = get_project_root()

    persist_dir = (
        project_root
        / "Naive RAG"
        / "data_base"
        / "vector_db"
        / "chroma"
    )

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding
    )

    return vectordb


def combine_docs(docs: List[Document]) -> str:
    """
    把 retriever 返回的 Document 列表合并成 prompt 里的 context 字符串。
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_retriever():
    """
    创建 hybrid retriever:
    Vector Retriever + BM25 Retriever + Ensemble Fusion + Cohere Rerank
    """
    texts = load_documents()
    split_text = split_documents(texts)

    vectordb = load_vectorstore()

    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": 10}
    )

    bm_retriever = BM25Retriever.from_documents(split_text)
    bm_retriever.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm_retriever],
        weights=[0.5, 0.5]
    )

    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=8
    )

    retriever = ContextualCompressionRetriever(
        base_retriever=hybrid_retriever,
        base_compressor=reranker
    )

    return retriever


def create_rag_chain():
    """
    创建支持 chat_history 的 RAG chain。

    输入格式：
    {
        "input": "用户问题",
        "chat_history": [
            ("human", "之前的问题"),
            ("ai", "之前的回答")
        ]
    }

    输出格式：
    {
        "input": ...,
        "chat_history": ...,
        "context": ...,
        "answer": ...
    }
    """
    retriever = create_retriever()

    llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0
    )

    condense_question_system_template = (
        "请根据聊天记录和用户最新问题，"
        "把用户最新问题改写成一个可以独立理解的问题。"
        "不要回答问题，只需要返回改写后的问题。"
    )

    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            RunnableLambda(lambda x: x["input"]) | retriever
        ),
        condense_question_prompt
        | llm
        | StrOutputParser()
        | retriever
    )

    system_prompt = (
        "你是一个问答任务的助手。"
        "请使用检索到的上下文片段回答问题。"
        "如果你不知道答案就说不知道。"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    qa_chain = (
        RunnablePassthrough.assign(
            context=lambda x: combine_docs(x["context"])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = (
        RunnablePassthrough.assign(
            context=retrieve_docs
        )
        .assign(
            answer=qa_chain
        )
    )

    return qa_history_chain