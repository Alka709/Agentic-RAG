from tavily import TavilyClient
from config import TAVILY_API_KEY

def create_web_search_client():
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not configured.")

    return TavilyClient(api_key=TAVILY_API_KEY)

def search_web(client,query,max_results=5):
    response=client.search(query=query,max_results=max_results)

    results=[]

    for item in response.get("results",[]):
        results.append({
            "title":item.get("title",""),
            "url":item.get("url",""),
            "content":item.get("content","")
        })
    return results