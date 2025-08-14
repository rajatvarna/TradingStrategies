import unittest
import json
from web_app.app import create_app
from web_app.extensions import db

class APITestCase(unittest.TestCase):
    def setUp(self):
        """Set up a test environment."""
        self.app = create_app()
        self.app.config.update({
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SECRET_KEY": "a-test-secret-key"
        })
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        """Tear down the test environment."""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_register_user(self):
        """Test user registration."""
        response = self.client.post('/api/register',
                                    data=json.dumps({
                                        'username': 'testuser',
                                        'email': 'test@example.com',
                                        'password': 'password123'
                                    }),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 201)
        self.assertIn('User registered successfully', response.get_data(as_text=True))

    def test_login_user(self):
        """Test user login."""
        # First, register a user
        self.client.post('/api/register',
                         data=json.dumps({
                             'username': 'testuser',
                             'email': 'test@example.com',
                             'password': 'password123'
                         }),
                         content_type='application/json')

        # Now, log in
        response = self.client.post('/api/login',
                                    data=json.dumps({
                                        'username': 'testuser',
                                        'password': 'password123'
                                    }),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn('token', json_data)

    def test_create_strategy_with_token(self):
        """Test creating a strategy with a valid token."""
        # Register and log in to get a token
        self.client.post('/api/register',
                         data=json.dumps({'username': 'testuser', 'email': 'test@example.com', 'password': 'password123'}),
                         content_type='application/json')
        login_response = self.client.post('/api/login',
                                          data=json.dumps({'username': 'testuser', 'password': 'password123'}),
                                          content_type='application/json')
        token = login_response.get_json()['token']

        # Create a strategy
        strategy_data = {
            'name': 'Test Strategy',
            'config': {
                'tickers': ['AAPL', 'GOOG'],
                'start_date': '2022-01-01',
                'end_date': '2023-01-01',
                'parameters': {}
            }
        }
        response = self.client.post('/api/strategies',
                                    headers={'Authorization': f'Bearer {token}'},
                                    data=json.dumps(strategy_data),
                                    content_type='application/json')

        self.assertEqual(response.status_code, 201)
        json_data = response.get_json()
        self.assertEqual(json_data['name'], 'Test Strategy')
        self.assertEqual(json_data['author'], 'testuser')

    def test_create_strategy_without_token(self):
        """Test that creating a strategy without a token fails."""
        strategy_data = {'name': 'Test Strategy', 'config': {}}
        response = self.client.post('/api/strategies',
                                    data=json.dumps(strategy_data),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 401)

if __name__ == '__main__':
    unittest.main()
