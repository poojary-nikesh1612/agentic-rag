from typing import Any

from dotenv import load_dotenv
from graph.state import GraphState
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", None)

    tavily_searchs = web_search_tool.invoke({"query": question})["results"]

    joined_result = "\n".join(
        [tavily_search["content"] for tavily_search in tavily_searchs]
    )

    doc = Document(page_content=joined_result)

    if documents is not None:
        documents.append(doc)
    else:
        documents = [doc]

    return {"documents": documents, "question": question}
