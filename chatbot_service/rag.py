import os
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- Database Connection ---
# The chatbot service needs to connect to the web app's database to access strategy information.
# We assume the services are run from the root of the project, so the path is relative.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_app', 'app.db')
DB_URI = f'sqlite:///{DB_PATH}'

class RAGSystem:
    """
    Encapsulates the entire logic for the Retrieval-Augmented Generation (RAG) system.
    This system powers the chatbot by grounding its responses in the information
    stored in the application's database (specifically, public trading strategies).

    The process involves:
    1. Loading strategy data from the database.
    2. Embedding this data into numerical vectors.
    3. Storing these vectors in a searchable index (FAISS).
    4. When a user asks a question, retrieving the most relevant vectors.
    5. Using the retrieved information to generate an answer.
    """
    def __init__(self):
        """
        Initializes the RAG system by loading the embedding model and building the vector store.
        """
        self.vector_store = None
        # Uses a pre-trained model from HuggingFace to convert text into embeddings.
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Kicks off the process to load data and build the vector store upon instantiation.
        self._load_and_build()

    def _load_documents_from_db(self):
        """
        Connects to the application's SQLite database, queries for public strategies,
        and converts them into a list of LangChain `Document` objects.
        Each document represents a single strategy.
        """
        if not os.path.exists(DB_PATH):
            print(f"Database file not found at {DB_PATH}. Skipping document loading.")
            return []

        engine = create_engine(DB_URI)
        documents = []
        try:
            with engine.connect() as connection:
                # The query selects public strategies that have a description, as this
                # forms the primary content for the RAG system.
                query = text("SELECT name, description, config_json FROM strategy WHERE is_public = 1 AND description IS NOT NULL")
                result = connection.execute(query)
                for row in result:
                    # Each strategy's details are combined into a single text block.
                    content = f"Strategy Name: {row[0]}\nDescription: {row[1]}\nConfiguration: {row[2]}"
                    # A LangChain Document is created with the content and metadata.
                    doc = Document(page_content=content, metadata={'source': row[0]})
                    documents.append(doc)
            print(f"Loaded {len(documents)} documents from the database.")
        except Exception as e:
            print(f"Error connecting to the database or loading documents: {e}")
        return documents

    def _load_and_build(self):
        """
        Orchestrates the creation of the knowledge base for the RAG system.
        It loads the documents from the DB and then uses them to build the FAISS vector store.
        FAISS is a library for efficient similarity search on dense vectors.
        """
        print("Loading documents and building vector store...")
        documents = self._load_documents_from_db()
        if documents:
            # This line converts the text documents into vectors and stores them in FAISS.
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Vector store built successfully.")
        else:
            print("No documents were loaded, so the vector store is empty.")

    def get_answer(self, question: str):
        """
        This is the main method for the RAG pipeline. It takes a user's question,
        retrieves the most relevant documents from the vector store, and then
        constructs an answer based on those documents.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated answer.
        """
        if not self.vector_store:
            return "The knowledge base is currently empty. I cannot answer questions."

        # 1. Retrieval: Use the vector store to find the most similar documents to the question.
        #    'k=2' means we retrieve the top 2 most relevant documents.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            return "I could not find any relevant information to answer your question."

        # 2. Generation (Mock): In a full RAG system, the retrieved documents would be
        #    fed into a language model (like GPT) along with the original question to generate
        #    a natural language answer. Here, we are mocking this step by simply formatting
        #    the content of the retrieved documents into a response.
        response_text = "I found some information that might help:\n\n"
        for i, doc in enumerate(relevant_docs):
            response_text += f"--- Source {i+1}: {doc.metadata.get('source', 'Unknown')} ---\n"
            response_text += f"{doc.page_content}\n\n"

        return response_text

# --- RAG System Factory (Singleton Pattern) ---
# This ensures that we only have one instance of the RAGSystem in the application.
# This is important because initializing the RAGSystem (loading models, building the
# vector store) is a computationally expensive operation.
_rag_system_instance = None

def get_rag_system():
    """
    Factory function to get a singleton instance of the RAGSystem.
    On the first call, it creates the RAGSystem instance. On subsequent calls,
    it returns the existing instance. This prevents re-loading the models and data.
    """
    global _rag_system_instance
    if _rag_system_instance is None:
        print("Creating new RAGSystem instance...")
        _rag_system_instance = RAGSystem()
    return _rag_system_instance
