import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


load_dotenv(find_dotenv(), override=True)


def get_project_root() -> Path:
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        )
    except Exception:
        return Path(__file__).resolve().parents[3]


def load_documents() -> List[Document]:
    folder_path = get_project_root() / "llm-universe" / "data_base" / "knowledge_db"
    loaders = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            suffix = file_path.rsplit(".", 1)[-1].lower()
            if suffix == "pdf":
                loaders.append(PyMuPDFLoader(file_path))
            elif suffix == "md":
                loaders.append(UnstructuredMarkdownLoader(file_path))

    documents: List[Document] = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)


def combine_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def merge_and_deduplicate(*doc_lists: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for docs in doc_lists:
        for doc in docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

    return unique_docs


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri="http://localhost:19530")


def build_milvus_collection(
    collection_name: str = "RAG_collection",
    force_rebuild: bool = False,
) -> None:
    client = get_milvus_client()
    embedding_model = get_embedding_model()
    split_docs = split_documents(load_documents())
    chunk_texts = [doc.page_content for doc in split_docs]

    if client.has_collection(collection_name=collection_name):
        if not force_rebuild:
            return
        client.drop_collection(collection_name=collection_name)

    dimension = len(embedding_model.embed_query("test"))
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )

    vectors = embedding_model.embed_documents(chunk_texts)
    data = [
        {
            "id": idx,
            "vector": vectors[idx],
            "text": chunk_texts[idx],
            "subject": "agent",
        }
        for idx in range(len(vectors))
    ]

    batch_size = 100
    for start in range(0, len(data), batch_size):
        client.insert(
            collection_name=collection_name,
            data=data[start : start + batch_size],
        )


def create_vector_retriever(
    collection_name: str = "RAG_collection",
    top_k: int = 10,
):
    client = get_milvus_client()
    embedding_model = get_embedding_model()

    if not client.has_collection(collection_name=collection_name):
        build_milvus_collection(collection_name=collection_name)

    def retrieve(query: str) -> List[Document]:
        query_vector = embedding_model.embed_query(query)
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "subject"],
        )

        docs = []
        for result in results[0]:
            entity = result["entity"]
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        "subject": entity.get("subject", ""),
                        "retriever": "milvus",
                        "score": result.get("distance"),
                    },
                )
            )
        return docs

    return retrieve


def create_bm25_retriever(top_k: int = 10) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(split_documents(load_documents()))
    retriever.k = top_k
    return retriever


def rerank_documents(query: str, candidate_docs: List[Document]) -> List[Document]:
    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=8,
    )
    return reranker.compress_documents(
        documents=candidate_docs,
        query=query,
    )


def create_hybrid_retriever(collection_name: str = "RAG_collection"):
    vector_retriever = create_vector_retriever(collection_name=collection_name)
    bm25_retriever = create_bm25_retriever()

    def retrieve(query: str) -> List[Document]:
        vector_docs = vector_retriever(query)
        bm25_docs = bm25_retriever.invoke(query)
        for doc in bm25_docs:
            doc.metadata["retriever"] = "bm25"

        candidate_docs = merge_and_deduplicate(vector_docs, bm25_docs)
        return rerank_documents(query, candidate_docs)

    return RunnableLambda(retrieve)


def create_rag_chain(collection_name: str = "RAG_collection"):
    retriever = create_hybrid_retriever(collection_name=collection_name)
    llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0,
    )

    condense_question_system_template = (
        "请根据聊天记录和用户最新问题，"
        "把用户最新问题改写成一个可以独立理解的问题。"
        "不要回答问题，只需要返回改写后的问题。"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    retrieve_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            RunnableLambda(lambda x: x["input"]) | retriever,
        ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。"
        "请使用检索到的上下文片段回答问题。"
        "如果你不知道答案就说不知道。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = (
        RunnablePassthrough.assign(context=lambda x: combine_docs(x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(context=retrieve_docs)
        .assign(answer=qa_chain)
    )


def search_local_knowledge(
    query: str,
    chat_history=None,
    collection_name: str = "RAG_collection",
) -> str:
    rag_chain = create_rag_chain(collection_name=collection_name)
    result = rag_chain.invoke(
        {
            "input": query,
            "chat_history": chat_history or [],
        }
    )
    return result["answer"]
