import os
import shutil
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv(find_dotenv())


def load_documents(folder_path: str):
    files_path = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            files_path.append(file_path)

    loaders = []

    for file_path in files_path:
        tag = file_path.split(".")[-1].lower()

        if tag == "pdf":
            loaders.append(PyMuPDFLoader(file_path))

        elif tag == "md":
            loaders.append(UnstructuredMarkdownLoader(file_path))

    texts = []

    for loader in loaders:
        texts.extend(loader.load())

    return texts


def build_vector_db():
    folder_path = "data_base/knowledge_db"
    persist_dir = "data_base/vector_db/chroma"

    texts = load_documents(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_text = text_splitter.split_documents(texts)

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_dir
    )

    batch_size = 50

    for i in range(0, len(split_text), batch_size):
        docs = split_text[i: i + batch_size]
        vectordb.add_documents(docs)

    vectordb.persist()

    print(f"向量数据库构建完成，共切分 {len(split_text)} 个 chunks")


if __name__ == "__main__":
    build_vector_db()