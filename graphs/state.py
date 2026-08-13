from typing import TypedDict,List,Dict,Any

class RAGState(TypedDict,total=False):
    question: str
    retrieved_documents: List[Dict[str,Any]]
    evaluation: Dict[str,Any]
    context: str
    answer: str