import unittest
from unittest.mock import patch, MagicMock

# Mock the Document class from langchain, as we don't need the real one
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

class TestRAGSystem(unittest.TestCase):
    """Unit tests for the RAGSystem class."""

    @patch('chatbot_service.rag.create_engine')
    @patch('chatbot_service.rag.FAISS')
    @patch('chatbot_service.rag.HuggingFaceEmbeddings')
    @patch('chatbot_service.rag.Document', MockDocument)
    def test_load_documents_from_db(self, mock_hf_embeddings, mock_faiss, mock_create_engine):
        """Test that documents are correctly loaded and formatted from the mock DB."""
        # 1. Setup mocks
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_db_result = [
            ("Strategy A", "Description A", '{"param": 1}'),
            ("Strategy B", "Description B", '{"param": 2}')
        ]
        mock_connection.execute.return_value = mock_db_result

        # 2. Import and instantiate the RAGSystem
        from chatbot_service.rag import RAGSystem
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()

        # 3. Assertions
        mock_create_engine.assert_called_once()
        mock_connection.execute.assert_called_once()
        self.assertIsNotNone(rag_system.vector_store)

        # Check that FAISS.from_documents was called correctly
        mock_faiss.from_documents.assert_called_once()
        args, kwargs = mock_faiss.from_documents.call_args
        documents = args[0]
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "Strategy Name: Strategy A\nDescription: Description A\nConfiguration: {\"param\": 1}")
        self.assertEqual(documents[0].metadata, {'source': 'Strategy A'})

    @patch('chatbot_service.rag.create_engine')
    @patch('chatbot_service.rag.FAISS')
    @patch('chatbot_service.rag.HuggingFaceEmbeddings')
    def test_get_answer(self, mock_hf_embeddings, mock_faiss, mock_create_engine):
        """Test the answer generation logic."""
        # 1. Setup mocks
        mock_create_engine.return_value.connect.return_value.__enter__.return_value.execute.return_value = [
            ("Test Strategy", "Test Description", '{}')
        ]

        # 2. Import and instantiate
        from chatbot_service.rag import RAGSystem
        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()

        # 3. Mock the retriever
        mock_retriever = MagicMock()
        retrieved_docs = [
            MockDocument(page_content="Content of relevant doc 1", metadata={'source': 'Doc1'}),
            MockDocument(page_content="Content of relevant doc 2", metadata={'source': 'Doc2'})
        ]
        mock_retriever.get_relevant_documents.return_value = retrieved_docs
        rag_system.vector_store.as_retriever.return_value = mock_retriever

        # 4. Call get_answer
        answer = rag_system.get_answer("Some question")

        # 5. Assertions
        rag_system.vector_store.as_retriever.assert_called_once()
        mock_retriever.get_relevant_documents.assert_called_once_with("Some question")
        self.assertIn("Content of relevant doc 1", answer)
        self.assertIn("--- Source 2: Doc2 ---", answer)

if __name__ == '__main__':
    unittest.main()
