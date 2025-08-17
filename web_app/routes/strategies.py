import json
from flask import Blueprint, request, jsonify, current_app
from web_app.models import db, Strategy, BacktestResult, User
from web_app.auth_decorators import token_required
from web_app.errors import ValidationError, ForbiddenError
from quant_strategies.strategy_blocks import CustomStrategy
from datetime import datetime, timedelta

# Blueprint for strategy-related endpoints
strategies_bp = Blueprint('strategies_bp', __name__)

def _validate_parameters(parameters):
    """
    A helper function to validate the structure of the 'parameters' object
    within a strategy's configuration. This ensures that custom strategies
    have well-formed parameters.
    """
    if not isinstance(parameters, dict):
        raise ValidationError("Parameters must be a dictionary.")

    # Check that each parameter has the required keys and a valid type.
    for name, definition in parameters.items():
        if not all(k in definition for k in ['type', 'value', 'default']):
            raise ValidationError(f"Parameter '{name}' is missing required keys ('type', 'value', 'default').")

        if definition['type'] not in ['int', 'float', 'str']:
            raise ValidationError(f"Parameter '{name}' has an invalid type.")

@strategies_bp.route('/api/strategies', methods=['POST'])
@token_required
def create_strategy(current_user):
    """
    Create a new strategy.
    The user must be authenticated. The request body must contain a 'name' and 'config'.
    This endpoint also enforces business logic, such as plan limits on the number
    of private strategies a user can create.
    ---
    (Flasgger docs omitted for brevity)
    """
    data = request.get_json()
    if not data or 'name' not in data or 'config' not in data:
        raise ValidationError('Missing name or config in request')

    is_public = data.get('is_public', True)

    # Business Logic: Enforce plan limits for private strategies.
    # This checks the user's current subscription plan and compares their
    # private strategy count against the plan's limit.
    if not is_public and current_user.plan:
        private_strategy_count = Strategy.query.filter_by(author=current_user, is_public=False).count()
        if private_strategy_count >= current_user.plan.private_strategies_limit:
            raise ForbiddenError(f'Your current plan allows for {current_user.plan.private_strategies_limit} private strategies. Please upgrade for more.')

    # Validate the structure of the configuration if it contains parameters.
    config_dict = data['config']
    if 'parameters' in config_dict:
        _validate_parameters(config_dict['parameters'])

    # Create and save the new strategy to the database.
    new_strategy = Strategy(
        name=data['name'],
        description=data.get('description', ''),
        category=data.get('category'),
        is_public=is_public,
        config_json=json.dumps(config_dict), # The config is stored as a JSON string.
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
    Ensures that the current user has permission to view the strategy
    (i.e., they are the author or the strategy is public).
    """
    strategy = Strategy.query.get_or_404(strategy_id)

    # Permission Check: A user can only view parameters for their own strategies or public ones.
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
    This endpoint demonstrates more complex query building with SQLAlchemy
    based on various query parameters.
    ---
    (Flasgger docs omitted for brevity)
    """
    # Base query for all public strategies.
    query = Strategy.query.filter_by(is_public=True)

    # --- Dynamic Filtering ---
    # Apply filters based on the query parameters provided in the request.
    if 'category' in request.args:
        query = query.filter(Strategy.category == request.args['category'])

    # For performance-based filters, we need to join with the BacktestResult table.
    if 'min_cagr' in request.args or 'max_drawdown' in request.args:
        query = query.join(BacktestResult)
        if 'min_cagr' in request.args:
            query = query.filter(BacktestResult.cagr >= float(request.args['min_cagr']))
        if 'max_drawdown' in request.args:
            query = query.filter(BacktestResult.max_drawdown <= float(request.args['max_drawdown']))

    # Filter for strategies that are downloadable by the current user (a premium feature).
    if request.args.get('downloadable', 'false').lower() == 'true':
        if not current_user.plan or current_user.plan.name != 'premium':
            return jsonify([]) # Return empty list if user is not premium

    # --- Dynamic Sorting ---
    # Apply sorting based on the 'sort_by' and 'order' query parameters.
    if 'sort_by' in request.args:
        sort_by = request.args['sort_by']
        order = request.args.get('order', 'desc').lower()

        # Join with BacktestResult if sorting by a performance metric.
        if sort_by in ['cagr', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']:
            query = query.join(BacktestResult)

        # Get the sorting attribute from the model class.
        sort_attr = getattr(BacktestResult, sort_by, None)
        if sort_attr is None:
            raise ValidationError(f"Invalid sort_by field: {sort_by}")

        # Apply the sorting direction.
        if order == 'asc':
            query = query.order_by(sort_attr.asc())
        else:
            query = query.order_by(sort_attr.desc())

    strategies = query.all()
    return jsonify([s.to_dict() for s in strategies])

@strategies_bp.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """
    Get a ranked leaderboard of public strategies.
    This is a public endpoint that showcases top-performing strategies.
    ---
    (Flasgger docs omitted for brevity)
    """
    sort_by = request.args.get('sort_by', 'sharpe_ratio', type=str)

    # Validate the sort_by field to prevent arbitrary column sorting.
    valid_sort_fields = ['cagr', 'volatility', 'sharpe_ratio', 'max_drawdown', 'alpha', 'beta']
    if sort_by not in valid_sort_fields:
        raise ValidationError(f'Invalid sort_by field. Must be one of {valid_sort_fields}')

    # Query and join Strategy with BacktestResult, filtering for public strategies
    # that have backtest results, and order them by the chosen metric.
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
    This is a premium feature, so it checks the user's subscription plan.
    ---
    (Flasgger docs omitted for brevity)
    """
    # Business Logic: This feature is only available to premium users.
    if not current_user.plan or current_user.plan.name != 'premium':
        raise ForbiddenError('This feature is only available to premium users.')

    original_strategy = Strategy.query.get_or_404(strategy_id)

    if not original_strategy.is_public:
        raise ForbiddenError('Only public strategies can be downloaded.')

    # Create a new private strategy for the current user, copying the config
    # from the original public strategy.
    new_strategy = Strategy(
        name=f"Copy of {original_strategy.name}",
        description=f"Copied from strategy #{original_strategy.id}. Original description: {original_strategy.description}",
        config_json=original_strategy.config_json,
        is_public=False,  # The copied strategy is always private
        author=current_user
    )
    db.session.add(new_strategy)
    db.session.commit()

    return jsonify(new_strategy.to_dict()), 201
