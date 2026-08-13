from typing import Any

from chains.generation import generation_chain
from state import GraphState


def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join([document["page_content"] for document in documents])
    generation = generation_chain.invoke({"question": question, "context": context})
    return {"documents": documents, "question": question, "generation": generation}
