def retrieval_router(state):
    evaluation=state["evaluation"]

    if evaluation.get("sufficient",False):
        return "answer"

    return "web_search"