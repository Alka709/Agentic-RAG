from langgraph.graph import StateGraph, START, END
from graphs.state import RAGState
from graphs.nodes import(
    retrieve_node,
    evaluate_node,
    answer_node,
    web_search_node
)
from graphs.router import retrieval_router

def build_rag_graph(llm,prompt,top_k):
    graph=StateGraph(RAGState)
    
    #nodes
    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            top_k
        )
    )

    graph.add_node(
        "evaluate",
        lambda state: evaluate_node(
            state,
            llm
        )
    )

    graph.add_node(
        "answer",
        lambda state: answer_node(
            state,
            llm,
            prompt
        )
    )

    graph.add_node(
        "web_search",
        lambda state: web_search_node(state)
    )

    #edges
    graph.add_edge(START,"retrieve")

    graph.add_edge("retrieve","evaluate")

    #conditional routing
    graph.add_conditional_edges(
        "evaluate",
        retrieval_router,
        {
            "answer":"answer",
            "web_search": "web_search"
        }
    )

    graph.add_edge(
        "web_search",
        "answer"
    )

    return graph.compile()