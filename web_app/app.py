import os
import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# --- App Initialization ---
app = Flask(__name__)

# --- Database Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Database Initialization ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- Database Models ---
class Strategy(db.Model):
    """
    Represents a trading strategy stored in the database.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    is_public = db.Column(db.Boolean, default=False, nullable=False)
    # The configuration for the CustomStrategy, stored as a JSON string
    config_json = db.Column(db.Text, nullable=False)

    backtest_result = db.relationship('BacktestResult', backref='strategy', uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        """
        Serializes the Strategy object to a dictionary.
        """
        result_dict = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'is_public': self.is_public,
            'config': json.loads(self.config_json),
            'backtest_result': self.backtest_result.to_dict() if self.backtest_result else None
        }
        return result_dict

class BacktestResult(db.Model):
    """
    Stores the performance metrics for a single backtest run.
    """
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False, unique=True)

    cagr = db.Column(db.Float, nullable=True)
    volatility = db.Column(db.Float, nullable=True)
    sharpe_ratio = db.Column(db.Float, nullable=True)
    max_drawdown = db.Column(db.Float, nullable=True)
    alpha = db.Column(db.Float, nullable=True)
    beta = db.Column(db.Float, nullable=True)

    run_at = db.Column(db.DateTime, server_default=db.func.now())

    def to_dict(self):
        return {
            'cagr': self.cagr,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'alpha': self.alpha,
            'beta': self.beta,
            'run_at': self.run_at.isoformat()
        }


# --- API Endpoints ---
@app.route('/api/strategies', methods=['POST'])
def create_strategy():
    """
    Creates a new strategy and saves it to the database.
    Expects a JSON payload with 'name', 'description', and 'config'.
    """
    data = request.get_json()
    if not data or not 'name' in data or not 'config' in data:
        return jsonify({'error': 'Missing name or config in request'}), 400

    try:
        # The config should be a valid dictionary for our CustomStrategy
        config_dict = data['config']

        new_strategy = Strategy(
            name=data['name'],
            description=data.get('description', ''),
            is_public=data.get('is_public', True), # Default to public for now
            config_json=json.dumps(config_dict)
        )
        db.session.add(new_strategy)
        db.session.commit()

        return jsonify(new_strategy.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def get_public_strategies():
    """
    Returns a list of all public strategies.
    """
    try:
        strategies = Strategy.query.filter_by(is_public=True).all()
        return jsonify([s.to_dict() for s in strategies]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """
    Returns a ranked list of public strategies based on a performance metric.
    Defaults to sorting by Sharpe Ratio.
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


if __name__ == '__main__':
    app.run(debug=True)
