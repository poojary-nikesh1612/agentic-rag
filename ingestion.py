from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.html import partition_html

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    output_dimensionality=1536,
    chunk_size=50,
    retry_min_seconds=10,
)
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs_list = []

# for url in urls:
#     elements = partition_html(
#         url=url,
#         chunking_strategy="basic",
#         max_characters=1_000_000,
#     )

#     for element in elements:
#         docs_list.append(
#             Document(
#                 page_content=element.text or "",
#                 metadata={
#                     "source": url,
#                     "category": str(element.category),
#                 },
#             )
#         )

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
# doc_splits = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings,
#     persist_directory="./.chroma",
# )

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()

if __name__ == "__main__":
    docs = retriever.invoke("Autonomous Agents")
    print(docs)
