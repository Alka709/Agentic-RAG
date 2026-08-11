def evaluate_retrieval(results,threshold=0.7):
    if not results:
        return{
            "relevance":0.0,
            "sufficient":False,
            "reason":"No document was retrieved"
        }
    scores=[
        result["score"]
        for result in results
    ]

    best_score=min(scores)
    average_score=sum(scores)/len(scores)

    # Count results that are reasonably close
    useful_chunks = sum(
        score <= distance_threshold
        for score in scores
    )

    # Basic heuristic
    sufficient = (
        best_score <= distance_threshold
        and useful_chunks >= 2
    )

    return {
        "sufficient": sufficient,
        "confidence": round(confidence, 3),
        "best_score": round(best_score, 3),
        "average_score": round(average_score, 3),
        "useful_chunks": useful_chunks,
        "reason": (
            "Retrieved context appears strong enough."
            if sufficient
            else "Retrieved context may be insufficient."
        )
    }