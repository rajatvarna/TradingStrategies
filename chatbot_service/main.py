from fastapi import FastAPI
from pydantic import BaseModel
from .ai_service import get_ai_service
import json

# Pydantic models for the request bodies
class ChatQuery(BaseModel):
    question: str
    user_id: str | None = None

class StrategyDescription(BaseModel):
    description: str

class StrategyConfig(BaseModel):
    config_json: str

# Initialize the FastAPI app
app = FastAPI(
    title="Quant Strategy AI Assistant",
    description="An AI-powered chatbot to answer questions and assist with strategy creation.",
    version="0.2.0",
)

@app.on_event("startup")
async def startup_event():
    """
    This function is called when the application starts.
    We can use it to ensure the AI service is loaded.
    """
    print("Application startup...")
    ai_service = get_ai_service()
    if ai_service.vector_store is None:
        print("RAG system component is not initialized. Check database connection and documents.")
    else:
        print("AI Service is ready.")

@app.get("/")
def read_root():
    """A simple endpoint to confirm the service is running."""
    return {"status": "Quant Assistant is running"}

@app.post("/chat")
def answer_question(query: ChatQuery):
    """
    The main endpoint for handling user questions (RAG-based).
    """
    print(f"Received query: {query.question}")
    ai_service = get_ai_service()
    answer = ai_service.get_answer(query.question)

    return {
        "question": query.question,
        "answer": answer
    }

@app.post("/generate_strategy_config")
async def generate_config(description: StrategyDescription):
    """
    Takes a natural language description of a trading strategy and returns a
    structured JSON configuration for it.
    """
    print(f"Received description for config generation: {description.description}")
    ai_service = get_ai_service()
    config_json = await ai_service.generate_config_from_text(description.description)

    return {"generated_config": config_json}

@app.post("/debug_strategy_config")
async def debug_config(config: StrategyConfig):
    """
    Takes a strategy configuration in JSON format and returns a debug analysis,
    including suggestions for improvement.
    """
    print(f"Received config for debugging: {config.config_json}")
    ai_service = get_ai_service()
    # The config might be a string, so parse it to a dict if necessary
    try:
        config_dict = json.loads(config.config_json)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in config_json."}

    analysis = await ai_service.debug_config(config_dict)

    return {"debug_analysis": analysis}

# To run this service locally:
# uvicorn chatbot_service.main:app --reload
