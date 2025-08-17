import unittest
import json
from unittest.mock import patch
from web_app.app import create_app
from web_app.extensions import db
from web_app.models import Strategy

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

    def test_create_and_get_api_keys(self):
        """Test creating and retrieving API keys."""
        # Register and log in to get a token
        self.client.post('/api/register',
                         data=json.dumps({'username': 'testuser_keys', 'email': 'keys@example.com', 'password': 'password123'}),
                         content_type='application/json')
        login_response = self.client.post('/api/login',
                                          data=json.dumps({'username': 'testuser_keys', 'password': 'password123'}),
                                          content_type='application/json')
        token = login_response.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        # Create an API key
        create_response = self.client.post('/api/me/api-keys', headers=headers)
        self.assertEqual(create_response.status_code, 201)
        create_data = create_response.get_json()
        self.assertIn('api_key', create_data)
        self.assertIn('This is the only time you will see the full key', create_data['message'])

        # Get the list of API keys
        get_response = self.client.get('/api/me/api-keys', headers=headers)
        self.assertEqual(get_response.status_code, 200)
        get_data = get_response.get_json()
        self.assertEqual(len(get_data), 1)
        self.assertIn('prefix', get_data[0])
        self.assertNotIn('key_hash', get_data[0]) # Ensure the full key/hash is not returned

    def test_portfolio_management(self):
        """Test creating a portfolio and adding a strategy to it."""
        # Register, log in, and create a strategy to work with
        self.client.post('/api/register',
                         data=json.dumps({'username': 'portfolio_user', 'email': 'portfolio@example.com', 'password': 'password123'}),
                         content_type='application/json')
        login_response = self.client.post('/api/login',
                                          data=json.dumps({'username': 'portfolio_user', 'password': 'password123'}),
                                          content_type='application/json')
        token = login_response.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        strategy_data = {'name': 'Portfolio Strategy', 'config': {'tickers': ['TSLA']}}
        strategy_response = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data), content_type='application/json')
        strategy_id = strategy_response.get_json()['id']

        # Create a new portfolio
        portfolio_data = {'name': 'My Test Portfolio'}
        create_portfolio_response = self.client.post('/api/portfolios', headers=headers, data=json.dumps(portfolio_data), content_type='application/json')
        self.assertEqual(create_portfolio_response.status_code, 201)
        portfolio_id = create_portfolio_response.get_json()['id']
        self.assertEqual(create_portfolio_response.get_json()['name'], 'My Test Portfolio')

        # Add the strategy to the portfolio
        add_strategy_response = self.client.post(f'/api/portfolios/{portfolio_id}/add_strategy', headers=headers, data=json.dumps({'strategy_id': strategy_id}), content_type='application/json')
        self.assertEqual(add_strategy_response.status_code, 200)
        self.assertIn(strategy_id, add_strategy_response.get_json()['strategy_ids'])

        # Get the portfolio to verify
        get_portfolio_response = self.client.get(f'/api/portfolios/{portfolio_id}', headers=headers)
        self.assertEqual(get_portfolio_response.status_code, 200)
        self.assertEqual(get_portfolio_response.get_json()['name'], 'My Test Portfolio')
        self.assertIn(strategy_id, get_portfolio_response.get_json()['strategy_ids'])

    @patch('web_app.routes.analysis.run_optimization')
    def test_optimize_strategy_endpoint(self, mock_run_optimization):
        """Test the optimization endpoint, mocking the actual analysis."""
        # Set up the mock to return a dummy result
        mock_run_optimization.return_value = {'param': 10, 'performance': 1.5}

        # Register a user
        self.client.post('/api/register',
                         data=json.dumps({'username': 'analysis_user', 'email': 'analysis@example.com', 'password': 'password123'}),
                         content_type='application/json')

        # Manually update the user's tier to premium
        with self.app.app_context():
            from web_app.models import User
            user = User.query.filter_by(username='analysis_user').first()
            user.tier = 'premium'
            db.session.commit()

        # Log in to get a token
        login_response = self.client.post('/api/login',
                                          data=json.dumps({'username': 'analysis_user', 'password': 'password123'}),
                                          content_type='application/json')
        token = login_response.get_json()['token']
        headers = {'Authorization': f'Bearer {token}'}

        # Create a strategy
        strategy_data = {'name': 'Analysis Strategy', 'config': {'tickers': ['NVDA']}}
        strategy_response = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data), content_type='application/json')
        strategy_id = strategy_response.get_json()['id']

        # Call the optimize endpoint
        optimize_params = {'parameter_name': 'period', 'start': 10, 'end': 20, 'step': 1}
        response = self.client.post(f'/api/strategies/{strategy_id}/optimize', headers=headers, data=json.dumps(optimize_params), content_type='application/json')

        # Assert that the endpoint was successful and returned the mock data
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'param': 10, 'performance': 1.5})

        # Assert that the mocked function was called
        mock_run_optimization.assert_called_once()

        # Inspect the arguments it was called with
        call_args, call_kwargs = mock_run_optimization.call_args
        called_strategy_obj = call_args[0]
        called_params = call_args[1]

        self.assertIsInstance(called_strategy_obj, Strategy)
        self.assertEqual(called_strategy_obj.id, strategy_id)
        self.assertEqual(called_params, optimize_params)

if __name__ == '__main__':
    unittest.main()
