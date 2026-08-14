from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello from agentic-rag!")
    res = app.invoke({"question": "Tell me the latest space news."})
    print(res['generation'])
