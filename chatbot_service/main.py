from fastapi import FastAPI
from pydantic import BaseModel

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

@app.get("/")
def read_root():
    """A simple endpoint to confirm the service is running."""
    return {"status": "Quant Assistant is running"}

@app.post("/chat")
def answer_question(query: ChatQuery):
    """
    The main endpoint for handling user questions.

    This is currently a placeholder. In the next phase, this endpoint will
    trigger the RAG workflow to generate a context-aware answer.
    """
    print(f"Received query: {query.question}")

    # Placeholder response
    response_text = "Hello, I am the Quant Assistant. My full capabilities are under development. How can I help you with our strategies today?"

    return {
        "question": query.question,
        "answer": response_text
    }

# To run this service locally:
# uvicorn chatbot_service.main:app --reload
