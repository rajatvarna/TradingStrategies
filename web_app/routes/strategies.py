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
    """
    data = request.get_json()
    if not data or not 'name' in data or not 'config' in data:
        return jsonify({'error': 'Missing name or config in request'}), 400

    try:
        config_dict = data['config']

        new_strategy = Strategy(
            name=data['name'],
            description=data.get('description', ''),
            is_public=data.get('is_public', True),
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
def get_public_strategies():
    """
    Get a list of all public strategies
    ---
    tags:
      - Strategies
    responses:
      200:
        description: A list of public strategies
        schema:
          type: array
          items:
            $ref: '#/definitions/Strategy'
    """
    try:
        strategies = Strategy.query.filter_by(is_public=True).all()
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
