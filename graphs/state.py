from typing import TypedDict,List,Dict,Any

class RAGState(TypedDict,total=False):
    question: str
    documents: list
    evaluation: Dict[str,Any]
    web_results: List[Dict[str,Any]]
    context: str
    answer: str