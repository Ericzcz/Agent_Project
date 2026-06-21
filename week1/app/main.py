from typing import List, Literal, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_chain import create_rag_chain


app = FastAPI(
    title="Naive RAG API",
    description="基于 FastAPI + LangChain + Chroma 的文档问答 API",
    version="1.0.0"
)

rag_chain = create_rag_chain()


class ChatMessage(BaseModel):
    role: Literal["human", "ai"]
    content: str


class AskRequest(BaseModel):
    input: str
    chat_history: List[ChatMessage] = []


class AskResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "Naive RAG API is running"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    chat_history = [
        (message.role, message.content)
        for message in request.chat_history
    ]

    result = rag_chain.invoke({
        "input": request.input,
        "chat_history": chat_history
    })

    return {
        "answer": result["answer"]
    }