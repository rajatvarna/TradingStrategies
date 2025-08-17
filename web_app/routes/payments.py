import stripe
from flask import Blueprint, request, jsonify, current_app
from web_app.models import db, User, Transaction
from web_app.auth_decorators import token_required
import os

payments_bp = Blueprint('payments_bp', __name__)

# It's recommended to set this in your environment variables
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

@payments_bp.route('/api/payments/create-checkout-session', methods=['POST'])
@token_required
def create_checkout_session(current_user):
    """
    Creates a Stripe Checkout session for subscriptions or donations.
    ---
    tags:
      - Payments
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - type
            - price_id
          properties:
            type:
              type: string
              description: The type of checkout session ('subscription' or 'donation').
              enum: ['subscription', 'donation']
            price_id:
              type: string
              description: The ID of the Stripe Price object.
            quantity:
              type: integer
              description: The quantity for the donation (in cents).
    responses:
      200:
        description: Stripe Checkout session created successfully.
      400:
        description: Bad request.
    """
    data = request.get_json()
    session_type = data.get('type')

    if session_type == 'subscription':
        price_id = data.get('price_id')
        if not price_id:
            return jsonify({'error': 'A price_id is required for subscriptions.'}), 400

        try:
            checkout_session = stripe.checkout.Session.create(
                line_items=[{'price': price_id, 'quantity': 1}],
                mode='subscription',
                success_url=os.getenv('FRONTEND_URL') + '/success?session_id={CHECKOUT_SESSION_ID}',
                cancel_url=os.getenv('FRONTEND_URL') + '/cancel',
                customer_email=current_user.email,
                client_reference_id=current_user.id
            )
            return jsonify({'url': checkout_session.url})
        except Exception as e:
            return jsonify(error=str(e)), 403

    elif session_type == 'donation':
        try:
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
        except Exception as e:
            return jsonify(error=str(e)), 403

    else:
        return jsonify({'error': 'Invalid session type specified.'}), 400

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
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        return 'Invalid signature', 400

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
