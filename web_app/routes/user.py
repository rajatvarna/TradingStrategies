from flask import Blueprint, request, jsonify
from web_app.models import db, User, APIKey, PaperAccount, Strategy
from web_app.auth_decorators import token_required, api_key_required
import secrets
import bcrypt
import json
from datetime import datetime, timedelta

user_bp = Blueprint('user_bp', __name__)

@user_bp.route('/api/me/api-keys', methods=['POST'])
@token_required
def create_api_key(current_user):
    """
    Generates a new API key for the authenticated user.
    """
    # Generate a new key and its prefix
    new_key = secrets.token_urlsafe(32)
    prefix = new_key[:8]

    # Hash the full key for storage
    key_hash = bcrypt.hashpw(new_key.encode('utf-8'), bcrypt.gensalt())

    api_key = APIKey(user_id=current_user.id, key_hash=key_hash, prefix=prefix)
    db.session.add(api_key)
    db.session.commit()

    # Return the full, unhashed key to the user ONCE.
    return jsonify({'api_key': new_key, 'message': 'Key generated successfully. This is the only time you will see the full key.'}), 201

@user_bp.route('/api/me/api-keys', methods=['GET'])
@token_required
def get_api_keys(current_user):
    """
    Lists all API keys (prefixes only) for the authenticated user.
    """
    keys = APIKey.query.filter_by(user_id=current_user.id).all()
    return jsonify([key.to_dict() for key in keys]), 200

@user_bp.route('/api/signals/<int:strategy_id>', methods=['GET'])
@api_key_required
def get_strategy_signals(current_user, strategy_id):
    """
    Returns the latest trading signal for a given strategy.
    Requires API key authentication and 'developer' tier access.
    """
    # 1. Tier-based access control
    if current_user.tier != 'developer':
        return jsonify({
            'error': 'Access denied. This feature requires a developer tier subscription.'
        }), 403

    strategy_obj = Strategy.query.get_or_404(strategy_id)

    try:
        # 2. Instantiate the strategy from its stored config
        from quant_strategies.strategy_blocks import CustomStrategy
        config = json.loads(strategy_obj.config_json)

        config['start_date'] = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        config['end_date'] = datetime.now().strftime('%Y-%m-%d')

        custom_strategy = CustomStrategy(config=config)

        # 3. Generate signals
        signals_data = custom_strategy.generate_signals()

        # 4. Extract the latest signal for each ticker
        latest_signals = {}
        for ticker, df in signals_data.items():
            if not df.empty:
                last_row = df.iloc[-1]
                latest_signals[ticker] = {
                    'date': last_row.name.strftime('%Y-%m-%d'),
                    'signal': int(last_row['Signal']), # -1, 0, or 1
                    'close_price': float(last_row['Close'])
                }

        return jsonify({
            'strategy_id': strategy_obj.id,
            'strategy_name': strategy_obj.name,
            'latest_signals': latest_signals
        }), 200

    except Exception as e:
        return jsonify({'error': 'Failed to generate signals.', 'details': str(e)}), 500

@user_bp.route('/api/strategies/<int:strategy_id>/deploy', methods=['POST'])
@token_required
def deploy_strategy(current_user, strategy_id):
    """
    Deploys or undeploys a strategy for paper trading.
    """
    strategy = Strategy.query.get_or_404(strategy_id)
    if strategy.author != current_user:
        return jsonify({'error': 'You can only deploy your own strategies.'}), 403

    # Create a paper account for the user if they don't have one
    if not current_user.paper_account:
        account = PaperAccount(user_id=current_user.id)
        db.session.add(account)

    # Toggle deployment status
    strategy.is_paper_deployed = not strategy.is_paper_deployed
    db.session.commit()

    status = "deployed" if strategy.is_paper_deployed else "undeployed"
    return jsonify({'message': f'Strategy {strategy.name} has been {status} for paper trading.'})

@user_bp.route('/api/me/paper/account', methods=['GET'])
@token_required
def get_paper_account(current_user):
    """
    Returns the status of the user's paper trading account.
    """
    account = current_user.paper_account
    if not account:
        return jsonify({'message': 'No paper trading account found. Deploy a strategy to create one.'}), 404

    positions = [{
        'symbol': p.symbol,
        'quantity': p.quantity,
        'average_entry_price': p.average_entry_price
    } for p in account.positions]

    return jsonify({
        'balance': account.balance,
        'buying_power': account.buying_power,
        'created_at': account.created_at.isoformat(),
        'positions': positions
    })
