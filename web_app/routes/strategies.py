import json
from flask import Blueprint, request, jsonify, current_app
from web_app.models import db, Strategy, BacktestResult, User
from web_app.auth_decorators import token_required
from web_app.errors import ValidationError, ForbiddenError
from quant_strategies.strategy_blocks import CustomStrategy
from datetime import datetime, timedelta

strategies_bp = Blueprint('strategies_bp', __name__)

def _validate_parameters(parameters):
    """
    Validates the structure of the parameters object.
    """
    if not isinstance(parameters, dict):
        raise ValidationError("Parameters must be a dictionary.")

    for name, definition in parameters.items():
        if not all(k in definition for k in ['type', 'value', 'default']):
            raise ValidationError(f"Parameter '{name}' is missing required keys ('type', 'value', 'default').")

        if definition['type'] not in ['int', 'float', 'str']:
            raise ValidationError(f"Parameter '{name}' has an invalid type.")

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
        raise ValidationError('Missing name or config in request')

    is_public = data.get('is_public', True)

    # Enforce plan limits for private strategies
    if not is_public and current_user.plan:
        private_strategy_count = Strategy.query.filter_by(author=current_user, is_public=False).count()
        if private_strategy_count >= current_user.plan.private_strategies_limit:
            raise ForbiddenError(f'Your current plan allows for {current_user.plan.private_strategies_limit} private strategies. Please upgrade for more.')

    config_dict = data['config']
    if 'parameters' in config_dict:
        _validate_parameters(config_dict['parameters'])

    new_strategy = Strategy(
        name=data['name'],
        description=data.get('description', ''),
        category=data.get('category'),
        is_public=is_public,
        config_json=json.dumps(config_dict),
        author=current_user
    )
    db.session.add(new_strategy)
    db.session.commit()

    return jsonify(new_strategy.to_dict()), 201

@strategies_bp.route('/api/strategies/<int:strategy_id>/parameters', methods=['GET'])
@token_required
def get_strategy_parameters(current_user, strategy_id):
    """
    Gets the parameters for a specific strategy.
    """
    strategy = Strategy.query.get_or_404(strategy_id)
    if strategy.author != current_user and not strategy.is_public:
        raise ForbiddenError("You don't have permission to view this strategy.")

    config = json.loads(strategy.config_json)
    parameters = config.get('parameters', {})

    return jsonify(parameters)

@strategies_bp.route('/api/strategies', methods=['GET'])
@token_required
def get_public_strategies(current_user):
    """
    Get a list of all public strategies with optional filtering and sorting.
    ---
    tags:
      - Strategies
    parameters:
      - in: query
        name: category
        type: string
        required: false
        description: Filter by strategy category.
      - in: query
        name: min_cagr
        type: number
        required: false
        description: Filter by minimum CAGR.
      - in: query
        name: max_drawdown
        type: number
        required: false
        description: Filter by maximum drawdown.
      - in: query
        name: downloadable
        type: boolean
        required: false
        description: Filter for strategies that are downloadable by the current user.
      - in: query
        name: sort_by
        type: string
        required: false
        description: The metric to sort by.
      - in: query
        name: order
        type: string
        required: false
        description: The sort order ('asc' or 'desc').
    responses:
      200:
        description: A list of public strategies.
    """
    query = Strategy.query.filter_by(is_public=True)

    # Filtering
    if 'category' in request.args:
        query = query.filter(Strategy.category == request.args['category'])

    if 'min_cagr' in request.args or 'max_drawdown' in request.args:
        query = query.join(BacktestResult)
        if 'min_cagr' in request.args:
            query = query.filter(BacktestResult.cagr >= float(request.args['min_cagr']))
        if 'max_drawdown' in request.args:
            query = query.filter(BacktestResult.max_drawdown <= float(request.args['max_drawdown']))

    if request.args.get('downloadable', 'false').lower() == 'true':
        if not current_user.plan or current_user.plan.name != 'premium':
            return jsonify([])

    # Sorting
    if 'sort_by' in request.args:
        sort_by = request.args['sort_by']
        order = request.args.get('order', 'desc').lower()

        # Join with BacktestResult if sorting by a metric
        if sort_by in ['cagr', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']:
            query = query.join(BacktestResult)

        sort_attr = getattr(BacktestResult, sort_by, None)
        if sort_attr is None:
            raise ValidationError(f"Invalid sort_by field: {sort_by}")

        if order == 'asc':
            query = query.order_by(sort_attr.asc())
        else:
            query = query.order_by(sort_attr.desc())

    strategies = query.all()
    return jsonify([s.to_dict() for s in strategies])

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
        raise ValidationError(f'Invalid sort_by field. Must be one of {valid_sort_fields}')

    # Query and join Strategy with BacktestResult, filtering for public strategies with results
    ranked_strategies = db.session.query(Strategy).join(BacktestResult).filter(
        Strategy.is_public == True
    ).order_by(
        getattr(BacktestResult, sort_by).desc()
    ).all()

    return jsonify([s.to_dict() for s in ranked_strategies]), 200

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
    if not current_user.plan or current_user.plan.name != 'premium':
        raise ForbiddenError('This feature is only available to premium users.')

    original_strategy = Strategy.query.get_or_404(strategy_id)

    if not original_strategy.is_public:
        raise ForbiddenError('Only public strategies can be downloaded.')

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
