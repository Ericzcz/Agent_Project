from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import run_agent, search_local_knowledge


app = FastAPI(
    title="Week3 Agent API",
    description="FastAPI interface for the week3 local RAG and tool-calling agent.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question to ask.")


class QueryResponse(BaseModel):
    answer: str
    mode: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/local-query", response_model=QueryResponse)
async def local_query(request: QueryRequest):
    answer = await run_in_threadpool(search_local_knowledge, request.query)
    return QueryResponse(answer=answer, mode="local")


@app.post("/agent-query", response_model=QueryResponse)
async def agent_query(request: QueryRequest):
    answer = await run_in_threadpool(run_agent, request.query)
    return QueryResponse(answer=answer, mode="agent")
