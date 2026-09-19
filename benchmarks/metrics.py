import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalizes text for robust token and substring matching."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def calculate_retrieval_metrics(
    retrieved_docs: List[Dict[str, Any]],
    ground_truth_item: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculates deterministic retrieval metrics using ground truth definitions:
    - context_precision: ratio of retrieved chunks that contain relevant information.
    - context_recall: proportion of required context items captured in retrieved chunks.
    - hit_rate: 1.0 if at least one relevant chunk is retrieved, else 0.0.
    - mrr: reciprocal rank of the first relevant chunk.
    """
    required_contexts = [
        normalize_text(rc) for rc in ground_truth_item.get("required_context", [])
    ]
    relevant_sources = [
        normalize_text(s) for s in ground_truth_item.get("relevant_sources", [])
    ]
    relevant_types = ground_truth_item.get("relevant_content_types", [])
    is_out_of_domain = ground_truth_item.get("expected_fallback", False)

    if is_out_of_domain:
        # For out-of-domain queries, internal docs are not relevant
        return {
            "context_precision": 0.0,
            "context_recall": 1.0 if not required_contexts else 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    if not retrieved_docs:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }

    # Identify which retrieved documents are relevant
    relevant_chunk_flags: List[bool] = []
    first_relevant_rank: Optional[int] = None

    for rank, doc in enumerate(retrieved_docs, 1):
        content_norm = normalize_text(doc.get("content", ""))
        source_norm = normalize_text(doc.get("source", ""))
        content_type = doc.get("content_type", "text")

        is_source_match = any(src in source_norm for src in relevant_sources) if relevant_sources else True
        is_type_match = content_type in relevant_types if relevant_types else True
        
        # Check if this document contains any required context snippet
        has_context_overlap = any(req in content_norm for req in required_contexts) if required_contexts else True

        is_relevant = (is_source_match and is_type_match and has_context_overlap)
        relevant_chunk_flags.append(is_relevant)

        if is_relevant and first_relevant_rank is None:
            first_relevant_rank = rank

    # 1. Context Precision = Relevant Chunks / Total Retrieved Chunks
    precision = sum(relevant_chunk_flags) / len(retrieved_docs) if retrieved_docs else 0.0

    # 2. Context Recall = Required contexts found in any retrieved document / Total required contexts
    if required_contexts:
        combined_retrieved_content = " ".join(
            normalize_text(doc.get("content", "")) for doc in retrieved_docs
        )
        matched_contexts = sum(
            1 for req in required_contexts if req in combined_retrieved_content
        )
        recall = matched_contexts / len(required_contexts)
    else:
        recall = 1.0 if precision > 0 else 0.0

    # 3. Hit Rate & MRR
    hit_rate = 1.0 if (first_relevant_rank is not None) else 0.0
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

    return {
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
    }


def calculate_generation_metrics(
    question: str,
    answer: str,
    context: str,
    ground_truth_answer: str,
    llm: Optional[Any] = None
) -> Dict[str, Optional[float]]:
    """
    Evaluates generation quality:
    - faithfulness: degree to which the answer is grounded in the retrieved context.
    - answer_relevance: alignment between the generated answer and ground truth / question.

    When running in deterministic/offline mode without an LLM (llm is None),
    generation metrics are not evaluated and return None.
    """
    if llm is None:
        return {"faithfulness": None, "answer_relevance": None}

    if not answer or not answer.strip():
        return {"faithfulness": 0.0, "answer_relevance": 0.0}

    # Run LLM judge
    try:
        from pydantic import BaseModel, Field
        from langchain_core.prompts import ChatPromptTemplate

        class QualityJudge(BaseModel):
            faithfulness: float = Field(
                description="Score 0.0 to 1.0 on whether the answer is fully supported by the context."
            )
            answer_relevance: float = Field(
                description="Score 0.0 to 1.0 on how well the answer addresses the question."
            )

        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an impartial RAG evaluator.
Evaluate the given answer on two dimensions:
1. Faithfulness (0.0 to 1.0): Is every claim in the answer directly supported by the context?
2. Answer Relevance (0.0 to 1.0): Does the answer directly and accurately address the question?

Output floating-point scores between 0.0 and 1.0."""),
            ("human", """Question: {question}
Ground Truth: {ground_truth}
Context: {context}
Generated Answer: {answer}""")
        ])

        structured_judge = llm.with_structured_output(QualityJudge)
        chain = judge_prompt | structured_judge
        result = chain.invoke({
            "question": question,
            "ground_truth": ground_truth_answer,
            "context": context[:3000],
            "answer": answer[:1500]
        })

        return {
            "faithfulness": round(float(result.faithfulness), 4),
            "answer_relevance": round(float(result.answer_relevance), 4)
        }
    except Exception:
        # Fallback to lexical metrics only if LLM call fails during LLM mode
        answer_tokens = set(re.findall(r"\w+", normalize_text(answer)))
        context_tokens = set(re.findall(r"\w+", normalize_text(context)))
        gt_tokens = set(re.findall(r"\w+", normalize_text(ground_truth_answer)))

        if not answer_tokens:
            return {"faithfulness": 0.0, "answer_relevance": 0.0}

        common_context = answer_tokens.intersection(context_tokens)
        lexical_faithfulness = len(common_context) / len(answer_tokens) if answer_tokens else 0.0

        common_gt = answer_tokens.intersection(gt_tokens)
        lexical_relevance = (2.0 * len(common_gt)) / (len(answer_tokens) + len(gt_tokens)) if (answer_tokens and gt_tokens) else 0.0

        return {
            "faithfulness": round(min(1.0, max(0.0, lexical_faithfulness)), 4),
            "answer_relevance": round(min(1.0, max(0.0, lexical_relevance)), 4)
        }
