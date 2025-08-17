import unittest
import json
from unittest.mock import patch, MagicMock
from web_app.app import create_app
from web_app.extensions import db
from web_app.models import User, Strategy
from .utils import get_auth_token

class PremiumFeaturesTestCase(unittest.TestCase):
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
            # Create users for different tiers
            free_user = User(username='freeuser', email='free@example.com', tier='free')
            free_user.set_password('password123')
            premium_user = User(username='premiumuser', email='premium@example.com', tier='premium')
            premium_user.set_password('password123')
            db.session.add_all([free_user, premium_user])
            db.session.commit()

    def tearDown(self):
        """Tear down the test environment."""
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_free_user_private_strategy_limit(self):
        """Test that a free user is limited to one private strategy."""
        token = get_auth_token(self.client, 'freeuser', 'password123')
        headers = {'Authorization': f'Bearer {token}'}

        # Create the first private strategy (should succeed)
        strategy_data_1 = {'name': 'Private Strategy 1', 'config': {}, 'is_public': False}
        response_1 = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data_1), content_type='application/json')
        self.assertEqual(response_1.status_code, 201)

        # Attempt to create a second private strategy (should fail)
        strategy_data_2 = {'name': 'Private Strategy 2', 'config': {}, 'is_public': False}
        response_2 = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data_2), content_type='application/json')
        self.assertEqual(response_2.status_code, 403)
        self.assertIn('Free tier users are limited to 1 private strategy', response_2.get_data(as_text=True))

    def test_premium_user_can_create_multiple_private_strategies(self):
        """Test that a premium user can create multiple private strategies."""
        token = get_auth_token(self.client, 'premiumuser', 'password123')
        headers = {'Authorization': f'Bearer {token}'}

        # Create the first private strategy
        strategy_data_1 = {'name': 'Premium Private 1', 'config': {}, 'is_public': False}
        response_1 = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data_1), content_type='application/json')
        self.assertEqual(response_1.status_code, 201)

        # Create the second private strategy
        strategy_data_2 = {'name': 'Premium Private 2', 'config': {}, 'is_public': False}
        response_2 = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data_2), content_type='application/json')
        self.assertEqual(response_2.status_code, 201)

    def test_free_user_cannot_access_premium_analysis(self):
        """Test that free users cannot access premium analysis endpoints."""
        token = get_auth_token(self.client, 'freeuser', 'password123')
        headers = {'Authorization': f'Bearer {token}'}

        # Create a strategy to analyze
        strategy_data = {'name': 'Test Strategy', 'config': {}}
        strategy_response = self.client.post('/api/strategies', headers=headers, data=json.dumps(strategy_data), content_type='application/json')
        strategy_id = strategy_response.get_json()['id']

        # Attempt to access premium endpoints
        optimize_params = {'parameter_name': 'period', 'start': 10, 'end': 20, 'step': 1}
        optimize_response = self.client.post(f'/api/strategies/{strategy_id}/optimize', headers=headers, data=json.dumps(optimize_params), content_type='application/json')
        self.assertEqual(optimize_response.status_code, 403)

        walkforward_params = {'parameter_name': 'period', 'start': 10, 'end': 20, 'step': 1, 'in_sample_days': 100, 'out_of_sample_days': 30}
        walkforward_response = self.client.post(f'/api/strategies/{strategy_id}/walkforward', headers=headers, data=json.dumps(walkforward_params), content_type='application/json')
        self.assertEqual(walkforward_response.status_code, 403)

    def test_free_user_cannot_download_strategy(self):
        """Test that a free user cannot download a public strategy."""
        free_token = get_auth_token(self.client, 'freeuser', 'password123')
        premium_token = get_auth_token(self.client, 'premiumuser', 'password123')

        # Premium user creates a public strategy
        public_strategy_data = {'name': 'Public Strategy', 'config': {}, 'is_public': True}
        strategy_response = self.client.post('/api/strategies', headers={'Authorization': f'Bearer {premium_token}'}, data=json.dumps(public_strategy_data), content_type='application/json')
        strategy_id = strategy_response.get_json()['id']

        # Free user attempts to download it
        response = self.client.post(f'/api/strategies/{strategy_id}/download', headers={'Authorization': f'Bearer {free_token}'})
        self.assertEqual(response.status_code, 403)
        self.assertIn('This feature is only available to premium users', response.get_data(as_text=True))

    @patch('stripe.checkout.Session.create')
    @patch.dict('os.environ', {
        'STRIPE_SECRET_KEY': 'sk_test_123',
        'FRONTEND_URL': 'http://localhost:3000'
    })
    def test_create_checkout_session(self, mock_stripe_checkout_create):
        """Test the creation of a Stripe Checkout session."""
        # Mock the Stripe API call
        mock_stripe_checkout_create.return_value = MagicMock(url='https://checkout.stripe.com/mock-url')

        token = get_auth_token(self.client, 'freeuser', 'password123')
        headers = {'Authorization': f'Bearer {token}'}

        # Test subscription session
        sub_data = {'type': 'subscription', 'price_id': 'price_12345'}
        sub_response = self.client.post('/api/payments/create-checkout-session', headers=headers, data=json.dumps(sub_data), content_type='application/json')
        self.assertEqual(sub_response.status_code, 200)
        self.assertEqual(sub_response.get_json()['url'], 'https://checkout.stripe.com/mock-url')

        # Test donation session
        don_data = {'type': 'donation', 'quantity': 1000} # $10.00
        don_response = self.client.post('/api/payments/create-checkout-session', headers=headers, data=json.dumps(don_data), content_type='application/json')
        self.assertEqual(don_response.status_code, 200)
        self.assertEqual(don_response.get_json()['url'], 'https://checkout.stripe.com/mock-url')

    def test_filter_downloadable_strategies(self):
        """Test filtering for downloadable strategies."""
        free_token = get_auth_token(self.client, 'freeuser', 'password123')
        premium_token = get_auth_token(self.client, 'premiumuser', 'password123')

        # Premium user creates a public strategy
        public_strategy_data = {'name': 'Public Strategy', 'config': {}, 'is_public': True}
        self.client.post('/api/strategies', headers={'Authorization': f'Bearer {premium_token}'}, data=json.dumps(public_strategy_data), content_type='application/json')

        # Free user filters for downloadable strategies (should get none)
        free_response = self.client.get('/api/strategies?downloadable=true', headers={'Authorization': f'Bearer {free_token}'})
        self.assertEqual(free_response.status_code, 200)
        self.assertEqual(len(free_response.get_json()), 0)

        # Premium user filters for downloadable strategies (should get one)
        premium_response = self.client.get('/api/strategies?downloadable=true', headers={'Authorization': f'Bearer {premium_token}'})
        self.assertEqual(premium_response.status_code, 200)
        self.assertEqual(len(premium_response.get_json()), 1)

if __name__ == '__main__':
    unittest.main()
