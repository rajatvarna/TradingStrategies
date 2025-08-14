from flask import Blueprint, request, jsonify
from web_app.models import db, Portfolio, Strategy
from web_app.auth_decorators import token_required

portfolios_bp = Blueprint('portfolios_bp', __name__)

@portfolios_bp.route('/api/portfolios', methods=['POST'])
@token_required
def create_portfolio(current_user):
    data = request.get_json()
    if not data or not 'name' in data:
        return jsonify({'error': 'Missing name for portfolio.'}), 400

    portfolio = Portfolio(name=data['name'], author=current_user)
    db.session.add(portfolio)
    db.session.commit()
    return jsonify(portfolio.to_dict()), 201

@portfolios_bp.route('/api/portfolios', methods=['GET'])
@token_required
def get_portfolios(current_user):
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    return jsonify([p.to_dict() for p in portfolios])

@portfolios_bp.route('/api/portfolios/<int:portfolio_id>', methods=['GET'])
@token_required
def get_portfolio(current_user, portfolio_id):
    portfolio = Portfolio.query.get_or_404(portfolio_id)
    if portfolio.author != current_user:
        return jsonify({'error': 'Not authorized to view this portfolio.'}), 403
    return jsonify(portfolio.to_dict())

@portfolios_bp.route('/api/portfolios/<int:portfolio_id>/add_strategy', methods=['POST'])
@token_required
def add_strategy_to_portfolio(current_user, portfolio_id):
    portfolio = Portfolio.query.get_or_404(portfolio_id)
    if portfolio.author != current_user:
        return jsonify({'error': 'Not authorized to modify this portfolio.'}), 403

    data = request.get_json()
    if not data or 'strategy_id' not in data:
        return jsonify({'error': 'Missing strategy_id.'}), 400

    strategy = Strategy.query.get_or_404(data['strategy_id'])
    if strategy.author != current_user:
        return jsonify({'error': 'You can only add your own strategies to a portfolio.'}), 403

    portfolio.strategies.append(strategy)
    db.session.commit()
    return jsonify(portfolio.to_dict())
