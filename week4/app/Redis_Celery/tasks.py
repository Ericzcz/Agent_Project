from app.Redis_Celery.celery_app import celery_app
from app.rag_chain import build_milvus_collection

@celery_app.task(bind=True)
def index_document(
    self, 
    collection_name: str = "RAG_collection", 
    force_rebuild: bool = False,
    embedding_batch_size: int = 64,
    embedding_max_concurrency: int = 3,
    ):
    self.update_state(
        state="PROGRESS",
        meta={"step": "building_index"},
    )

    build_milvus_collection(
        batch_size=embedding_batch_size,
        max_concurrency=embedding_max_concurrency,
        collection_name=collection_name,
        force_rebuild=force_rebuild,
    )

    return {
        "status": "index",
        "collection_name": collection_name,
    }