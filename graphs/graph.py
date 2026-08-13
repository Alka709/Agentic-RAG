from langgraph.graph import StateGraph, START, END
from graphs.state import RAGState
from graphs.nodes import(
    retrieve_node,
    evaluate_node,
    answer_node
)
from graphs.router import retrieval_router

def build_rag_graph(vector_store,llm,prompt,top_k):
    graph=StateGraph(RAGState)
    
    #nodes
    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            vector_store,
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

    #edges
    graph.add_edge(START,"retrieve")

    graph.add_edge("retrieve","evaluate")

    #conditional routing
    graph.add_conditional_edges(
        "evaluate",
        retrieval_router,
        {
            "answer":"answer",
            "web_search":END
        }
    )

    graph.add_edge(
        "answer",
        END
    )

    return graph.compile()