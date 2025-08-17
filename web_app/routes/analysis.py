from flask import Blueprint, request, jsonify
from web_app.models import db, Portfolio, Strategy
from web_app.auth_decorators import token_required
from optimizer import run_optimization
from portfolio_backtester import PortfolioBacktester
from walk_forward_analyzer import run_walk_forward_analysis

analysis_bp = Blueprint('analysis_bp', __name__)

@analysis_bp.route('/api/portfolios/<int:portfolio_id>/backtest', methods=['POST'])
@token_required
def backtest_portfolio(current_user, portfolio_id):
    """
    Runs a backtest on a full portfolio of strategies.
    """
    portfolio = Portfolio.query.get_or_404(portfolio_id)
    if portfolio.author != current_user:
        return jsonify({'error': 'Not authorized to backtest this portfolio.'}), 403

    try:
        backtester = PortfolioBacktester(portfolio_obj=portfolio)
        results = backtester.run()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': 'An error occurred during portfolio backtest.', 'details': str(e)}), 500

@analysis_bp.route('/api/strategies/<int:strategy_id>/walkforward', methods=['POST'])
@token_required
def walk_forward_strategy(current_user, strategy_id):
    """
    Runs a walk-forward analysis for a given strategy.
    """
    strategy = Strategy.query.get_or_404(strategy_id)
    if strategy.author != current_user:
        return jsonify({'error': 'You can only analyze your own strategies.'}), 403

    data = request.get_json()
    opt_fields = ['parameter_name', 'start', 'end', 'step']
    wf_fields = ['in_sample_days', 'out_of_sample_days']
    if not data or not all(field in data for field in opt_fields + wf_fields):
        return jsonify({'error': f'Missing one or more required fields: {opt_fields + wf_fields}'}), 400

    try:
        optimization_params = {k: data[k] for k in opt_fields}
        in_sample_days = data['in_sample_days']
        out_of_sample_days = data['out_of_sample_days']

        results = run_walk_forward_analysis(strategy, optimization_params, in_sample_days, out_of_sample_days)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': 'An error occurred during walk-forward analysis.', 'details': str(e)}), 500

@analysis_bp.route('/api/strategies/<int:strategy_id>/optimize', methods=['POST'])
@token_required
def optimize_strategy(current_user, strategy_id):
    """
    Runs parameter optimization for a given strategy.
    """
    strategy = Strategy.query.get_or_404(strategy_id)
    if strategy.author != current_user:
        return jsonify({'error': 'You can only optimize your own strategies.'}), 403

    data = request.get_json()
    required_fields = ['parameter_name', 'start', 'end', 'step']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': f'Missing one or more required fields: {required_fields}'}), 400

    try:
        results = run_optimization(strategy, data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': 'An error occurred during optimization.', 'details': str(e)}), 500
