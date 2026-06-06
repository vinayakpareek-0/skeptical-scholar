from functools import lru_cache

from config import PROJECT_ROOT, load_config
from ingestion.database import get_all_chunks as load_chunks
from ingestion.database import init_db


@lru_cache(maxsize=1)
def get_config():
    return load_config()


@lru_cache(maxsize=1)
def get_chunks():
    config = get_config()
    conn = init_db(PROJECT_ROOT / config["database"]["path"])
    try:
        return load_chunks(conn)
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_bm25_index():
    from retrieval.bm25_retriever import build_bm25_index

    bm25_index, chunks = build_bm25_index(get_chunks())
    return bm25_index, chunks


@lru_cache(maxsize=1)
def get_dense_index():
    from retrieval.dense_retriever import load_index as load_dense_index

    config = get_config()
    return load_dense_index(PROJECT_ROOT / config["dense"]["index_path"])


@lru_cache(maxsize=1)
def get_dense_model():
    from retrieval.dense_retriever import load_dense_model

    config = get_config()
    return load_dense_model(config["dense"]["model_name"])


@lru_cache(maxsize=1)
def get_reranker():
    from retrieval.reranker import load_reranker

    config = get_config()
    return load_reranker(config["retrieval"]["reranker_name"])


@lru_cache(maxsize=1)
def get_entity_extractor():
    from reasoning.entity_extract import load_extractor

    return load_extractor()


@lru_cache(maxsize=1)
def get_nli_model():
    from reasoning.contradiction_detect import load_nli_model

    return load_nli_model()


@lru_cache(maxsize=1)
def get_llm_client():
    from generation.llm_client import load_llm

    return load_llm()


def clear_runtime_cache():
    for cached_fn in (
        get_config,
        get_chunks,
        get_bm25_index,
        get_dense_index,
        get_dense_model,
        get_reranker,
        get_entity_extractor,
        get_nli_model,
        get_llm_client,
    ):
        cached_fn.cache_clear()
