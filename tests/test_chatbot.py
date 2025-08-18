import unittest
from unittest.mock import patch, MagicMock
import json
import asyncio

# Mock the Document class from langchain
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Mock the pipeline for generative AI tests
class MockGenerator:
    def __init__(self, return_value):
        self.return_value = return_value
    def __call__(self, prompt, max_new_tokens):
        return [{'generated_text': self.return_value}]

class TestAIService(unittest.TestCase):
    """Unit tests for the AIService class."""

    @patch('chatbot_service.ai_service.create_engine')
    @patch('chatbot_service.ai_service.FAISS')
    @patch('chatbot_service.ai_service.HuggingFaceEmbeddings')
    @patch('chatbot_service.ai_service.pipeline')
    @patch('chatbot_service.ai_service.Document', MockDocument)
    def test_rag_document_loading(self, mock_pipeline, mock_hf_embeddings, mock_faiss, mock_create_engine):
        """Test that documents are correctly loaded for the RAG system."""
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_db_result = [("Strategy A", "Description A", '{"param": 1}')]
        mock_connection.execute.return_value = mock_db_result

        from chatbot_service.ai_service import AIService
        with patch('os.path.exists', return_value=True):
            ai_service = AIService()

        mock_faiss.from_documents.assert_called_once()
        args, _ = mock_faiss.from_documents.call_args
        self.assertEqual(len(args[0]), 1)
        self.assertEqual(args[0][0].page_content, "Strategy Name: Strategy A\nDescription: Description A\nConfiguration: {\"param\": 1}")

    @patch('chatbot_service.ai_service.AIService')
    def _test_generate_config_from_text_async(self, MockAIService):
        """Test the natural language to JSON generation feature."""
        # Setup mock AI service and its generator
        mock_ai_service = MockAIService.return_value
        mock_response = """
        Here is the JSON you requested:
        JSON Configuration:
        {
            "name": "Moving Average Crossover",
            "tickers": ["SPY"],
            "parameters": {
                "short_window": 20,
                "long_window": 50
            }
        }
        """
        mock_ai_service.generator = MockGenerator(mock_response)

        async def mock_async_generate(description):
            # This is a simplified version of the actual method for testing purposes
            generated_text = mock_ai_service.generator(description, max_new_tokens=150)[0]['generated_text']
            json_str = generated_text.split("JSON Configuration:")[-1].strip()
            return json.loads(json_str)

        mock_ai_service.generate_config_from_text.side_effect = mock_async_generate

        # Call the async method
        description = "Create a simple moving average crossover strategy for SPY with a 20-day and 50-day window."
        result = asyncio.run(mock_ai_service.generate_config_from_text(description))

        # Assertions
        self.assertIn("name", result)
        self.assertEqual(result["name"], "Moving Average Crossover")
        self.assertEqual(result["parameters"]["short_window"], 20)

    def test_generate_config(self):
        asyncio.run(self._test_generate_config_from_text_async())


    @patch('chatbot_service.ai_service.AIService')
    def _test_debug_config_async(self, MockAIService):
        """Test the configuration debugging feature."""
        # Setup mock AI service
        mock_ai_service = MockAIService.return_value
        mock_response = """
        Your Analysis:
        The configuration looks mostly correct. However, the 'ema_period' parameter
        is set to 200, which is very long for a short-term breakout strategy.
        Consider lowering it to a value between 10 and 50.
        """
        mock_ai_service.generator = MockGenerator(mock_response)

        async def mock_async_debug(config):
            config_str = json.dumps(config, indent=4)
            prompt = f"Debug this: {config_str}" # Simplified prompt
            generated_text = mock_ai_service.generator(prompt, max_new_tokens=200)[0]['generated_text']
            return generated_text.split("Your Analysis:")[-1].strip()

        mock_ai_service.debug_config.side_effect = mock_async_debug

        # Call the async method
        config_to_debug = {"name": "Test", "parameters": {"ema_period": 200}}
        result = asyncio.run(mock_ai_service.debug_config(config_to_debug))

        # Assertions
        self.assertIn("The configuration looks mostly correct", result)
        self.assertIn("lowering it to a value between 10 and 50", result)

    def test_debug_config(self):
        asyncio.run(self._test_debug_config_async())

if __name__ == '__main__':
    unittest.main()
