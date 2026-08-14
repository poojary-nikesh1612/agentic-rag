from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello from agentic-rag!")
    res = app.invoke({"question": "what is agent memory?"})
    print(res['generation'])
