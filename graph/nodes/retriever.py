from typing import Any

from ingestion import retriever
from state import GraphState


def retriever_node(state: GraphState) -> dict[str, Any]:
    print("--RETRIEVER--")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
