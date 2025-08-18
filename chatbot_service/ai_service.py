import os
import json
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from transformers import pipeline, set_seed

# --- Database Connection ---
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_app', 'app.db')
DB_URI = f'sqlite:///{DB_PATH}'

class AIService:
    """
    Encapsulates the AI logic for the chatbot, including RAG for Q&A,
    and LLM-based generation and debugging of strategies.
    """
    def __init__(self):
        """
        Initializes the AI service by loading the embedding model, the vector store,
        and the generative language model.
        """
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._load_and_build_rag()

        # Load a generative model for text-to-json and debugging tasks
        # Using a smaller, instruction-following model is ideal for local performance.
        # Note: This will download the model on the first run.
        print("Loading generative language model...")
        self.generator = pipeline('text-generation', model='HuggingFaceH4/zephyr-7b-beta')
        print("Generative language model loaded.")


    def _load_documents_from_db(self):
        """
        Connects to the database and loads public strategies for the RAG system.
        """
        if not os.path.exists(DB_PATH):
            print(f"Database file not found at {DB_PATH}. Skipping document loading.")
            return []

        engine = create_engine(DB_URI)
        documents = []
        try:
            with engine.connect() as connection:
                query = text("SELECT name, description, config_json FROM strategy WHERE is_public = 1 AND description IS NOT NULL")
                result = connection.execute(query)
                for row in result:
                    content = f"Strategy Name: {row[0]}\nDescription: {row[1]}\nConfiguration: {row[2]}"
                    doc = Document(page_content=content, metadata={'source': row[0]})
                    documents.append(doc)
            print(f"Loaded {len(documents)} documents from the database.")
        except Exception as e:
            print(f"Error connecting to the database or loading documents: {e}")
        return documents

    def _load_and_build_rag(self):
        """
        Builds the FAISS vector store for the RAG system.
        """
        print("Loading documents and building vector store for RAG...")
        documents = self._load_documents_from_db()
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Vector store built successfully.")
        else:
            print("No documents were loaded, so the vector store is empty.")

    def get_answer(self, question: str):
        """
        RAG pipeline: Retrieves relevant documents and returns their content.
        """
        if not self.vector_store:
            return "The knowledge base is currently empty. I cannot answer questions."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            return "I could not find any relevant information to answer your question."

        response_text = "I found some information that might help:\n\n"
        for i, doc in enumerate(relevant_docs):
            response_text += f"--- Source {i+1}: {doc.metadata.get('source', 'Unknown')} ---\n"
            response_text += f"{doc.page_content}\n\n"

        return response_text

    async def generate_config_from_text(self, description: str):
        """
        Uses the generative LLM to create a strategy config from a natural language description.
        """
        prompt = f"""
        You are an expert in quantitative trading strategies. Your task is to convert a natural language
        description of a trading strategy into a valid JSON configuration for our backtesting system.

        Here is an example of the target JSON format:
        {{
            "name": "My Crypto Momentum Strategy",
            "tickers": ["BTC-USD", "ETH-USD"],
            "start_date": "2022-01-01",
            "end_date": "2023-01-01",
            "parameters": {{
                "hh_period": 20,
                "ema_period": 5,
                "risk_on_btc_ema_period": 20,
                "min_price": 0.50,
                "min_volume": 100000
            }}
        }}

        Now, based on the following description, generate the JSON configuration.
        Only output the JSON object, with no other text or explanation.

        Description: "{description}"
        JSON Configuration:
        \"\"\"

        try:
            # The model might return more than just the JSON, so we need to parse it.
            generated_text = self.generator(prompt, max_new_tokens=150)[0]['generated_text']
            # Extract the JSON part from the generated text
            json_str = generated_text.split("JSON Configuration:")[-1].strip()
            config_dict = json.loads(json_str)
            return config_dict
        except Exception as e:
            print(f"Error during config generation: {e}")
            return {"error": "Failed to generate a valid configuration from the description."}


    async def debug_config(self, config: dict):
        """
        Uses the generative LLM to analyze a strategy configuration and provide feedback.
        """
        config_str = json.dumps(config, indent=4)
        prompt = f"""
        You are an expert in quantitative trading strategies and you are debugging a strategy configuration
        for a user. Analyze the following JSON configuration and provide feedback.

        Check for:
        1.  Logical errors (e.g., conflicting parameters).
        2.  Potential improvements (e.g., better parameter choices, missing filters).
        3.  Clarity and correctness of the configuration structure.

        Provide your feedback as a concise analysis.

        Configuration to debug:
        ```json
        {config_str}
        ```

        Your Analysis:
        \"\"\"

        try:
            generated_text = self.generator(prompt, max_new_tokens=200)[0]['generated_text']
            # Extract the analysis part
            analysis = generated_text.split("Your Analysis:")[-1].strip()
            return analysis
        except Exception as e:
            print(f"Error during config debugging: {e}")
            return {"error": "Failed to analyze the configuration."}


# --- AI Service Factory (Singleton Pattern) ---
_ai_service_instance = None

def get_ai_service():
    """
    Factory function to get a singleton instance of the AIService.
    """
    global _ai_service_instance
    if _ai_service_instance is None:
        print("Creating new AIService instance...")
        _ai_service_instance = AIService()
    return _ai_service_instance
