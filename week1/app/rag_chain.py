from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv(find_dotenv())


def combine(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    persist_dir = "data_base/vector_db/chroma"

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

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

    retriever_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            RunnableLambda(lambda x: x["input"]) | retriever
        ),
        condense_question_prompt | llm | StrOutputParser() | retriever
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
            context=lambda x: combine(x["context"])
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = (
        RunnablePassthrough.assign(
            context=retriever_docs
        ).assign(
            answer=qa_chain
        )
    )

    return qa_history_chain