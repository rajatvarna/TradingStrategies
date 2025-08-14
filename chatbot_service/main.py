from fastapi import FastAPI
from pydantic import BaseModel
from .rag import rag_system

# Pydantic model for the request body
class ChatQuery(BaseModel):
    question: str
    user_id: str | None = None # Optional user ID for context

# Initialize the FastAPI app
app = FastAPI(
    title="Quant Strategy AI Assistant",
    description="An AI-powered chatbot to answer questions about our trading strategies.",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    """
    This function is called when the application starts.
    We can use it to ensure the RAG system is loaded.
    """
    print("Application startup...")
    if rag_system.vector_store is None:
        print("RAG system is not initialized. Check database connection and documents.")
    else:
        print("RAG system is ready.")


@app.get("/")
def read_root():
    """A simple endpoint to confirm the service is running."""
    return {"status": "Quant Assistant is running"}

@app.post("/chat")
def answer_question(query: ChatQuery):
    """
    The main endpoint for handling user questions.
    This endpoint triggers the RAG workflow to generate a context-aware answer.
    """
    print(f"Received query: {query.question}")

    answer = rag_system.get_answer(query.question)

    return {
        "question": query.question,
        "answer": answer
    }

# To run this service locally:
# uvicorn chatbot_service.main:app --reload
