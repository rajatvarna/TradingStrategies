import unittest
from unittest.mock import patch, MagicMock

# Mock the Document class from langchain, as we don't need the real one
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Since rag.py will be imported by the test, we need to patch the dependencies
# BEFORE the import happens. This is a common pattern when testing code with
# heavy dependencies or side effects on import.
@patch('chatbot_service.rag.FAISS', MagicMock())
@patch('chatbot_service.rag.HuggingFaceEmbeddings', MagicMock())
@patch('chatbot_service.rag.Document', MockDocument)
class TestRAGSystem(unittest.TestCase):
    """Unit tests for the RAGSystem class."""

    @patch('chatbot_service.rag.create_engine')
    def test_load_documents_from_db(self, mock_create_engine):
        """Test that documents are correctly loaded and formatted from the mock DB."""
        # 1. Setup the mock database engine and connection
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        # 2. Define the mock data that the database query should return
        mock_db_result = [
            ("Strategy A", "Description A", '{"param": 1}'),
            ("Strategy B", "Description B", '{"param": 2}')
        ]
        mock_connection.execute.return_value = mock_db_result

        # 3. Instantiate the RAGSystem (this will call _load_documents_from_db)
        # We patch 'os.path.exists' to simulate the DB file being present.
        with patch('os.path.exists', return_value=True):
            from chatbot_service.rag import RAGSystem, FAISS
            rag_system = RAGSystem()

        # 4. Assertions
        # Check that the database was connected to
        mock_create_engine.assert_called_once()
        mock_connection.execute.assert_called_once()

        # The vector store should be built because we returned documents
        self.assertIsNotNone(rag_system.vector_store)

        # Check that FAISS.from_documents was called with the correct Document objects
        # This is a bit complex, but we can check the number of documents.
        FAISS.from_documents.assert_called_once()
        args, kwargs = FAISS.from_documents.call_args
        documents = args[0]
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].page_content, "Strategy Name: Strategy A\nDescription: Description A\nConfiguration: {\"param\": 1}")
        self.assertEqual(documents[0].metadata, {'source': 'Strategy A'})

    @patch('chatbot_service.rag.create_engine')
    def test_get_answer(self, mock_create_engine):
        """Test the answer generation logic."""
        # 1. Setup mocks
        mock_create_engine.return_value.connect.return_value.__enter__.return_value.execute.return_value = [
            ("Test Strategy", "Test Description", '{}')
        ]

        with patch('os.path.exists', return_value=True):
            rag_system = RAGSystem()

        # 2. Mock the retriever's response
        mock_retriever = MagicMock()
        retrieved_docs = [
            MockDocument(page_content="Content of relevant doc 1", metadata={'source': 'Doc1'}),
            MockDocument(page_content="Content of relevant doc 2", metadata={'source': 'Doc2'})
        ]
        mock_retriever.get_relevant_documents.return_value = retrieved_docs
        rag_system.vector_store.as_retriever.return_value = mock_retriever

        # 3. Call get_answer
        answer = rag_system.get_answer("Some question")

        # 4. Assertions
        rag_system.vector_store.as_retriever.assert_called_once()
        mock_retriever.get_relevant_documents.assert_called_once_with("Some question")
        self.assertIn("Content of relevant doc 1", answer)
        self.assertIn("--- Source 2: Doc2 ---", answer)

if __name__ == '__main__':
    unittest.main()
