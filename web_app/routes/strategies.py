import json
from flask import Blueprint, request, jsonify, current_app
from web_app.models import db, Strategy, BacktestResult, User
from web_app.auth_decorators import token_required
from quant_strategies.strategy_blocks import CustomStrategy
from datetime import datetime, timedelta

strategies_bp = Blueprint('strategies_bp', __name__)

@strategies_bp.route('/api/strategies', methods=['POST'])
@token_required
def create_strategy(current_user):
    """
    Create a new strategy
    ---
    tags:
      - Strategies
    security:
      - Bearer: []
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - name
            - config
          properties:
            name:
              type: string
              description: The name of the strategy.
            description:
              type: string
              description: A description of the strategy.
            is_public:
              type: boolean
              description: Whether the strategy should be publicly visible.
            config:
              type: object
              description: The configuration object for the strategy.
    responses:
      201:
        description: Strategy created successfully
      400:
        description: Bad request (e.g., missing fields)
      401:
        description: Unauthorized (invalid or missing token)
      403:
        description: Forbidden (e.g., tier limit reached)
    """
    data = request.get_json()
    if not data or not 'name' in data or not 'config' in data:
        return jsonify({'error': 'Missing name or config in request'}), 400

    is_public = data.get('is_public', True)

    # Enforce tier limits for private strategies
    if not is_public and current_user.tier == 'free':
        private_strategy_count = Strategy.query.filter_by(author=current_user, is_public=False).count()
        if private_strategy_count >= 1:
            return jsonify({'error': 'Free tier users are limited to 1 private strategy. Please upgrade to premium.'}), 403

    try:
        config_dict = data['config']

        new_strategy = Strategy(
            name=data['name'],
            description=data.get('description', ''),
            is_public=is_public,
            config_json=json.dumps(config_dict),
            author=current_user
        )
        db.session.add(new_strategy)
        db.session.commit()

        return jsonify(new_strategy.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@strategies_bp.route('/api/strategies', methods=['GET'])
@token_required
def get_public_strategies(current_user):
    """
    Get a list of all public strategies
    ---
    tags:
      - Strategies
    parameters:
      - in: query
        name: downloadable
        type: boolean
        required: false
        description: Filter for strategies that are downloadable by the current user.
    responses:
      200:
        description: A list of public strategies
        schema:
          type: array
          items:
            $ref: '#/definitions/Strategy'
    """
    downloadable = request.args.get('downloadable', 'false').lower() == 'true'

    try:
        query = Strategy.query.filter_by(is_public=True)

        if downloadable:
            if current_user.tier != 'premium':
                return jsonify([]) # Non-premium users cannot download any strategies
            # If the user is premium, all public strategies are considered downloadable

        strategies = query.all()
        return jsonify([s.to_dict() for s in strategies]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@strategies_bp.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """
    Get a ranked leaderboard of public strategies
    ---
    tags:
      - Strategies
    parameters:
      - in: query
        name: sort_by
        type: string
        required: false
        default: sharpe_ratio
        enum: ['cagr', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']
        description: The metric to sort the leaderboard by.
    responses:
      200:
        description: A ranked list of public strategies
        schema:
          type: array
          items:
            $ref: '#/definitions/Strategy'
      400:
        description: Invalid sort_by field
    """
    sort_by = request.args.get('sort_by', 'sharpe_ratio', type=str)

    valid_sort_fields = ['cagr', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']
    if sort_by not in valid_sort_fields:
        return jsonify({'error': f'Invalid sort_by field. Must be one of {valid_sort_fields}'}), 400

    try:
        # Query and join Strategy with BacktestResult, filtering for public strategies with results
        ranked_strategies = db.session.query(Strategy).join(BacktestResult).filter(
            Strategy.is_public == True
        ).order_by(
            getattr(BacktestResult, sort_by).desc()
        ).all()

        return jsonify([s.to_dict() for s in ranked_strategies]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@strategies_bp.route('/api/strategies/<int:strategy_id>/download', methods=['POST'])
@token_required
def download_strategy(current_user, strategy_id):
    """
    Downloads (copies) a public strategy to the user's account as a private strategy.
    ---
    tags:
      - Strategies
    security:
      - Bearer: []
    parameters:
      - name: strategy_id
        in: path
        type: integer
        required: true
        description: The ID of the public strategy to download.
    responses:
      201:
        description: Strategy copied successfully.
      403:
        description: Forbidden (user is not premium or strategy is not public).
      404:
        description: Strategy not found.
    """
    if current_user.tier != 'premium':
        return jsonify({'error': 'This feature is only available to premium users.'}), 403

    original_strategy = Strategy.query.get_or_404(strategy_id)

    if not original_strategy.is_public:
        return jsonify({'error': 'Only public strategies can be downloaded.'}), 403

    try:
        # Create a new private strategy for the current user
        new_strategy = Strategy(
            name=f"Copy of {original_strategy.name}",
            description=f"Copied from strategy #{original_strategy.id}. Original description: {original_strategy.description}",
            config_json=original_strategy.config_json,
            is_public=False,  # The copied strategy is private
            author=current_user
        )
        db.session.add(new_strategy)
        db.session.commit()

        return jsonify(new_strategy.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
