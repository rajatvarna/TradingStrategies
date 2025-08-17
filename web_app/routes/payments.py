import stripe
import os
from flask import Blueprint, request, jsonify, current_app
from web_app.models import db, User, Transaction
from web_app.auth_decorators import token_required
from web_app.errors import ValidationError

payments_bp = Blueprint('payments_bp', __name__)

@payments_bp.route('/api/payments/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(current_user):
    """
    Creates a Stripe Checkout session for subscriptions or donations.
    """
    data = request.get_json()
    session_type = data.get('type')
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

    if session_type == 'subscription':
        price_id = data.get('price_id')
        if not price_id:
            raise ValidationError('A price_id is required for subscriptions.')

        checkout_session = stripe.checkout.Session.create(
            line_items=[{'price': price_id, 'quantity': 1}],
            mode='subscription',
            success_url=os.getenv('FRONTEND_URL') + '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=os.getenv('FRONTEND_URL') + '/cancel',
            customer_email=current_user.email,
            client_reference_id=current_user.id
        )
        return jsonify({'url': checkout_session.url})

    elif session_type == 'donation':
        checkout_session = stripe.checkout.Session.create(
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {'name': 'Donation'},
                    'unit_amount': data.get('quantity', 500),  # Default to $5.00
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=os.getenv('FRONTEND_URL') + '/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=os.getenv('FRONTEND_URL') + '/cancel',
            customer_email=current_user.email,
            client_reference_id=current_user.id
        )
        return jsonify({'url': checkout_session.url})

    else:
        raise ValidationError('Invalid session type specified.')

@payments_bp.route('/api/payments/webhook', methods=['POST'])
def stripe_webhook():
    """
    Stripe webhook endpoint to handle payment events.
    """
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        raise ValidationError('Invalid payload')
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        raise ValidationError('Invalid signature')

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('client_reference_id')
        user = User.query.get(user_id)

        if user:
            if session.mode == 'subscription':
                # Logic to update user's tier to 'premium'
                user.tier = 'premium'
                transaction_type = 'subscription'
            elif session.mode == 'payment':
                # This was a one-time donation
                transaction_type = 'donation'

            # Create a transaction record
            new_transaction = Transaction(
                user_id=user.id,
                transaction_type=transaction_type,
                amount=session.amount_total / 100.0,  # Amount is in cents
                stripe_charge_id=session.payment_intent
            )
            db.session.add(new_transaction)
            db.session.commit()

    return 'Success', 200
