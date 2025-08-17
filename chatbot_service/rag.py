import os
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- Database Connection ---
# The chatbot service needs to connect to the web app's database.
# We assume the services are run from the root of the project, so the path is relative.
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'web_app', 'app.db')
DB_URI = f'sqlite:///{DB_PATH}'

class RAGSystem:
    """
    Encapsulates the logic for the Retrieval-Augmented Generation system.
    """
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._load_and_build()

    def _load_documents_from_db(self):
        """
        Loads public strategies from the database and converts them to LangChain Documents.
        """
        if not os.path.exists(DB_PATH):
            print(f"Database file not found at {DB_PATH}. Skipping document loading.")
            return []

        engine = create_engine(DB_URI)
        documents = []
        try:
            with engine.connect() as connection:
                # We query for public strategies that have a description
                query = text("SELECT name, description, config_json FROM strategy WHERE is_public = 1 AND description IS NOT NULL")
                result = connection.execute(query)
                for row in result:
                    # Combine the strategy info into a single text content
                    content = f"Strategy Name: {row[0]}\nDescription: {row[1]}\nConfiguration: {row[2]}"
                    doc = Document(page_content=content, metadata={'source': row[0]})
                    documents.append(doc)
            print(f"Loaded {len(documents)} documents from the database.")
        except Exception as e:
            print(f"Error connecting to the database or loading documents: {e}")
        return documents

    def _load_and_build(self):
        """
        Loads the documents and builds the FAISS vector store.
        """
        print("Loading documents and building vector store...")
        documents = self._load_documents_from_db()
        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("Vector store built successfully.")
        else:
            print("No documents were loaded, so the vector store is empty.")

    def get_answer(self, question: str):
        """
        Retrieves relevant documents and generates a simple answer.
        This mocks the "generation" part of RAG.
        """
        if not self.vector_store:
            return "The knowledge base is currently empty. I cannot answer questions."

        # 1. Retrieve relevant documents
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2}) # Get top 2 results
        relevant_docs = retriever.get_relevant_documents(question)

        if not relevant_docs:
            return "I could not find any relevant information to answer your question."

        # 2. Mock Generation: Create a response from the retrieved documents
        response_text = "I found some information that might help:\n\n"
        for i, doc in enumerate(relevant_docs):
            response_text += f"--- Source {i+1}: {doc.metadata.get('source', 'Unknown')} ---\n"
            response_text += f"{doc.page_content}\n\n"

        return response_text

# --- RAG System Factory ---
_rag_system_instance = None

def get_rag_system():
    """
    Factory function to get a singleton instance of the RAGSystem.
    This makes it easier to manage the instance and mock it for tests.
    """
    global _rag_system_instance
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem()
    return _rag_system_instance
