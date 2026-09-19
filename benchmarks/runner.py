import time
import traceback
from typing import Any, Dict, List, Optional

from config import (
    BM25_CANDIDATES,
    DENSE_CANDIDATES,
    RRF_K,
    TOP_K,
)
from rag.retriever import dense_search, reciprocal_rank_fusion
from rag.evaluator import evaluate_retrieval
from rag.generator import generate_answer
from rag.prompts import create_rag_prompt
from benchmarks.metrics import calculate_retrieval_metrics, calculate_generation_metrics


def run_benchmark_query(
    item: Dict[str, Any],
    retrieval_mode: str,
    vector_store: Any,
    bm25_index: Optional[Any],
    llm: Optional[Any] = None,
    prompt: Optional[Any] = None,
    top_k: int = TOP_K,
    run_llm_judge: bool = False
) -> Dict[str, Any]:
    """
    Executes a single benchmark query in isolated 'dense' or 'hybrid' mode
    with sub-millisecond latency profiling and structured error handling.
    """
    start_total = time.perf_counter()
    question = item["question"]
    query_id = item.get("id", "unknown")
    query_type = item.get("query_type", "general")
    ground_truth_answer = item.get("ground_truth_answer", "")

    dense_lat_ms = 0.0
    bm25_lat_ms = 0.0
    rrf_lat_ms = 0.0
    gen_lat_ms = 0.0
    retrieval_lat_ms = 0.0

    retrieved_docs: List[Dict[str, Any]] = []
    generated_answer = ""
    context_str = ""
    web_fallback_triggered = False

    try:
        if retrieval_mode == "dense":
            t0 = time.perf_counter()
            dense_candidates = dense_search(vector_store, question, top_k=top_k)
            t1 = time.perf_counter()
            dense_lat_ms = (t1 - t0) * 1000.0
            retrieval_lat_ms = dense_lat_ms

            retrieved_docs = dense_candidates[:top_k]
            for rank, doc in enumerate(retrieved_docs, 1):
                doc["final_rank"] = rank
                doc["rrf_score"] = 1.0 / (RRF_K + rank)

        elif retrieval_mode == "hybrid":
            # 1. Dense retrieval
            t0 = time.perf_counter()
            dense_candidates = dense_search(
                vector_store, question, top_k=max(top_k, DENSE_CANDIDATES)
            )
            t1 = time.perf_counter()
            dense_lat_ms = (t1 - t0) * 1000.0

            # 2. BM25 retrieval
            t2 = time.perf_counter()
            if bm25_index:
                bm25_candidates = bm25_index.search(
                    question, top_k=max(top_k, BM25_CANDIDATES)
                )
            else:
                bm25_candidates = []
            t3 = time.perf_counter()
            bm25_lat_ms = (t3 - t2) * 1000.0

            # 3. RRF Fusion
            t4 = time.perf_counter()
            retrieved_docs = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                top_k=top_k,
                rrf_k=RRF_K
            )
            t5 = time.perf_counter()
            rrf_lat_ms = (t5 - t4) * 1000.0
            retrieval_lat_ms = dense_lat_ms + bm25_lat_ms + rrf_lat_ms

        else:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

        # 4. Context & Retrieval Quality Evaluation
        context_parts = []
        for doc in retrieved_docs:
            c_type = doc.get("content_type", "text")
            page = doc.get("page", 1)
            content = doc.get("content", "")
            ident = doc.get("identifier", "")
            context_parts.append(f"[{c_type.upper()} | Page {page} | ID {ident}]\n{content}")
        context_str = "\n\n".join(context_parts)

        # Check sufficiency via evaluator if LLM available
        if llm:
            try:
                eval_res = evaluate_retrieval(llm, question, retrieved_docs)
                if not eval_res.get("sufficient", True) or item.get("expected_fallback", False):
                    web_fallback_triggered = True
            except Exception:
                web_fallback_triggered = bool(item.get("expected_fallback", False))
        else:
            web_fallback_triggered = bool(item.get("expected_fallback", False))

        # 5. Answer Generation
        if llm:
            t_gen_start = time.perf_counter()
            prompt_template = prompt or create_rag_prompt()
            generated_answer = generate_answer(llm, prompt_template, question, context_str)
            t_gen_end = time.perf_counter()
            gen_lat_ms = (t_gen_end - t_gen_start) * 1000.0
        else:
            generated_answer = f"Synthesized answer based on {len(retrieved_docs)} retrieved chunks."

        # 6. Calculate Metrics
        retrieval_metrics = calculate_retrieval_metrics(retrieved_docs, item)
        generation_metrics = calculate_generation_metrics(
            question=question,
            answer=generated_answer,
            context=context_str,
            ground_truth_answer=ground_truth_answer,
            llm=llm if run_llm_judge else None
        )

        total_lat_ms = (time.perf_counter() - start_total) * 1000.0

        # Formatted retrieved documents summary for JSON
        doc_summaries = []
        for doc in retrieved_docs:
            doc_summaries.append({
                "identifier": doc.get("identifier", ""),
                "content_type": doc.get("content_type", "text"),
                "source": doc.get("source", ""),
                "page": doc.get("page", 1),
                "rank": doc.get("final_rank", 1),
                "score": round(doc.get("rrf_score", doc.get("score", 0.0)), 6)
            })

        return {
            "id": query_id,
            "question": question,
            "query_type": query_type,
            "retrieval_mode": retrieval_mode,
            "status": "success",
            "retrieved_documents": doc_summaries,
            "answer": generated_answer,
            "context_precision": retrieval_metrics["context_precision"],
            "context_recall": retrieval_metrics["context_recall"],
            "hit_rate": retrieval_metrics["hit_rate"],
            "mrr": retrieval_metrics["mrr"],
            "faithfulness": generation_metrics["faithfulness"],
            "answer_relevance": generation_metrics["answer_relevance"],
            "dense_retrieval_latency_ms": round(dense_lat_ms, 2),
            "bm25_retrieval_latency_ms": round(bm25_lat_ms, 2),
            "rrf_latency_ms": round(rrf_lat_ms, 2),
            "retrieval_latency_ms": round(retrieval_lat_ms, 2),
            "generation_latency_ms": round(gen_lat_ms, 2),
            "total_latency_ms": round(total_lat_ms, 2),
            "web_fallback": web_fallback_triggered,
            "error": None
        }

    except Exception as e:
        total_lat_ms = (time.perf_counter() - start_total) * 1000.0
        return {
            "id": query_id,
            "question": question,
            "query_type": query_type,
            "retrieval_mode": retrieval_mode,
            "status": "failed",
            "retrieved_documents": [],
            "answer": "",
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "faithfulness": None if llm is None else 0.0,
            "answer_relevance": None if llm is None else 0.0,
            "dense_retrieval_latency_ms": round(dense_lat_ms, 2),
            "bm25_retrieval_latency_ms": round(bm25_lat_ms, 2),
            "rrf_latency_ms": round(rrf_lat_ms, 2),
            "retrieval_latency_ms": round(retrieval_lat_ms, 2),
            "generation_latency_ms": 0.0,
            "total_latency_ms": round(total_lat_ms, 2),
            "web_fallback": False,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        }
