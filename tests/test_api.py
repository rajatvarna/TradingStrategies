import unittest
import json
from unittest.mock import patch
from web_app.app import create_app
from web_app.extensions import db
from web_app.models import Strategy
from tests.utils import get_auth_token

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
        token = get_auth_token(self.client, 'testuser', 'password123')

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
        token = get_auth_token(self.client, 'testuser_keys', 'password123')
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
        token = get_auth_token(self.client, 'portfolio_user', 'password123')
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

        # Register, log in, and create a strategy
        self.client.post('/api/register',
                         data=json.dumps({'username': 'analysis_user', 'email': 'analysis@example.com', 'password': 'password123'}),
                         content_type='application/json')
        with self.app.app_context():
            from web_app.models import User
            user = User.query.filter_by(username='analysis_user').first()
            user.tier = 'premium'
            db.session.commit()
        token = get_auth_token(self.client, 'analysis_user', 'password123')
        headers = {'Authorization': f'Bearer {token}'}
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

    def test_filter_and_sort_strategies(self):
        """Test filtering and sorting public strategies."""
        # Register and log in to get a token
        self.client.post('/api/register',
                         data=json.dumps({'username': 'filter_user', 'email': 'filter@example.com', 'password': 'password123'}),
                         content_type='application/json')
        token = get_auth_token(self.client, 'filter_user', 'password123')
        headers = {'Authorization': f'Bearer {token}'}

        # Create some strategies with different categories and backtest results
        with self.app.app_context():
            from web_app.models import User, BacktestResult
            user = User.query.filter_by(username='filter_user').first()
            s1 = Strategy(name='S1', category='mean-reversion', is_public=True, config_json='{}', author=user)
            s2 = Strategy(name='S2', category='trend-following', is_public=True, config_json='{}', author=user)
            s3 = Strategy(name='S3', category='mean-reversion', is_public=True, config_json='{}', author=user)
            db.session.add_all([s1, s2, s3])
            db.session.commit()
            br1 = BacktestResult(strategy_id=s1.id, cagr=0.1, sharpe_ratio=1.2)
            br2 = BacktestResult(strategy_id=s2.id, cagr=0.2, sharpe_ratio=1.5)
            br3 = BacktestResult(strategy_id=s3.id, cagr=0.05, sharpe_ratio=0.8)
            db.session.add_all([br1, br2, br3])
            db.session.commit()

        # Test filtering by category
        response = self.client.get('/api/strategies?category=mean-reversion', headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['name'], 'S1')
        self.assertEqual(data[1]['name'], 'S3')

        # Test filtering by min_cagr
        response = self.client.get('/api/strategies?min_cagr=0.15', headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'S2')

        # Test sorting by sharpe_ratio
        response = self.client.get('/api/strategies?sort_by=sharpe_ratio', headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]['name'], 'S2')
        self.assertEqual(data[1]['name'], 'S1')
        self.assertEqual(data[2]['name'], 'S3')

if __name__ == '__main__':
    unittest.main()
