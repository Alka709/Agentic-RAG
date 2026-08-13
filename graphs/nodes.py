from rag.retriever import retrieve_documents
from rag.evaluator import evaluate_retrieval
from rag.generator import generate_answer

def retrieve_node(state,vector_store,top_k):
    question = state["question"]
    
    results=retrieve_documents(
        vector_store,
        question,
        top_k
    )

    return{
        "retrieved_documents":results
    }

def evaluate_node(state,llm):
    question=state["question"]
    results=state["retrieved_documents"]

    evaluation=evaluate_retrieval(
        llm,
        question,results
    )

    return{
        "evaluation": evaluation
    }

def answer_node(state,llm,prompt):
    question=state["question"]
    results=state["retrieved_documents"]

    context="\n\n".join(
        result["content"]
        for result in results
    )

    answer=generate_answer(
        llm,
        prompt,
        question,
        context
    )

    return{
        "context": context,
        "answer": answer
    }