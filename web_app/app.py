import os
import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import bcrypt
import jwt
import secrets
from datetime import datetime, timedelta, timezone

# --- Path Setup for Quant Lib ---
# This is needed so the web app can import from the quant_strategies library
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quant_strategies.strategy_blocks import CustomStrategy
# The 'from optimizer' import is moved inside the endpoint to prevent circular dependency

# --- App Initialization ---
app = Flask(__name__)

# --- Database Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'a-super-secret-key-that-should-be-changed' # Change in production

# --- Database Initialization ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- Database Models ---
class User(db.Model):
    """
    Represents a user of the application.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    tier = db.Column(db.String(32), default='free', nullable=False) # e.g., 'free', 'premium', 'developer'

    strategies = db.relationship('Strategy', backref='author', lazy='dynamic')
    api_keys = db.relationship('APIKey', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    paper_account = db.relationship('PaperAccount', backref='user', uselist=False, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def __repr__(self):
        return f'<User {self.username}>'

class APIKey(db.Model):
    """
    Stores API keys for users.
    """
    id = db.Column(db.Integer, primary_key=True)
    key_hash = db.Column(db.String(128), nullable=False)
    prefix = db.Column(db.String(8), unique=True, nullable=False) # For identification
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    def to_dict(self):
        return {
            'id': self.id,
            'prefix': self.prefix,
            'created_at': self.created_at.isoformat()
        }

class Strategy(db.Model):
    """
    Represents a trading strategy stored in the database.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    is_public = db.Column(db.Boolean, default=False, nullable=False)
    is_paper_deployed = db.Column(db.Boolean, default=False, nullable=False)
    # The configuration for the CustomStrategy, stored as a JSON string
    config_json = db.Column(db.Text, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    backtest_result = db.relationship('BacktestResult', backref='strategy', uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        """
        Serializes the Strategy object to a dictionary.
        """
        result_dict = {
            'id': self.id,
            'name': self.name,
            'author': self.author.username if self.author else 'Anonymous',
            'description': self.description,
            'is_public': self.is_public,
            'is_paper_deployed': self.is_paper_deployed,
            'config': json.loads(self.config_json),
            'backtest_result': self.backtest_result.to_dict() if self.backtest_result else None
        }
        return result_dict

class PaperAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    balance = db.Column(db.Float, nullable=False, default=100000.0)
    buying_power = db.Column(db.Float, nullable=False, default=100000.0)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    positions = db.relationship('PaperPosition', backref='account', lazy='dynamic', cascade="all, delete-orphan")
    trades = db.relationship('PaperTrade', backref='account', lazy='dynamic', cascade="all, delete-orphan")

class PaperPosition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('paper_account.id'), nullable=False)
    symbol = db.Column(db.String(32), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    average_entry_price = db.Column(db.Float, nullable=False)

class PaperTrade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('paper_account.id'), nullable=False)
    symbol = db.Column(db.String(32), nullable=False)
    action = db.Column(db.String(16), nullable=False) # 'BUY' or 'SELL'
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

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


from functools import wraps

# --- Auth Decorators ---
def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'message': 'API key is missing!'}), 401

        # Find key by prefix
        prefix = api_key[:8]
        key_record = APIKey.query.filter_by(prefix=prefix).first()

        if key_record and bcrypt.checkpw(api_key.encode('utf-8'), key_record.key_hash):
            current_user = User.query.get(key_record.user_id)
            return f(current_user, *args, **kwargs)

        return jsonify({'message': 'API key is invalid!'}), 401
    return decorated

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] # Bearer <token>

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401

        return f(current_user, *args, **kwargs)
    return decorated

# --- API Endpoints ---
@app.route('/api/register', methods=['POST'])
def register():
    """
    Registers a new user.
    """
    data = request.get_json()
    if not data or not 'username' in data or not 'password' in data or not 'email' in data:
        return jsonify({'error': 'Missing username, email, or password'}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email address already in use'}), 400

    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    """
    Authenticates a user and returns a JWT.
    """
    data = request.get_json()
    if not data or not 'username' in data or not 'password' in data:
        return jsonify({'error': 'Missing username or password'}), 400

    user = User.query.filter_by(username=data['username']).first()

    if user and user.check_password(data['password']):
        # Create the token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.now(timezone.utc) + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')

        return jsonify({'token': token})

    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/api/strategies', methods=['POST'])
@token_required
def create_strategy(current_user):
    """
    Creates a new strategy and saves it to the database.
    Requires authentication.
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

@app.route('/api/me/api-keys', methods=['POST'])
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

@app.route('/api/me/api-keys', methods=['GET'])
@token_required
def get_api_keys(current_user):
    """
    Lists all API keys (prefixes only) for the authenticated user.
    """
    keys = APIKey.query.filter_by(user_id=current_user.id).all()
    return jsonify([key.to_dict() for key in keys]), 200

@app.route('/api/signals/<int:strategy_id>', methods=['GET'])
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
        config = json.loads(strategy_obj.config_json)

        # Ensure the date range is just the last few days to get the latest signal
        # without a full backtest. We need enough data for the longest indicator.
        # This is a simplification; a real system might need more robust date logic.
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


# --- Paper Trading Endpoints ---

@app.route('/api/strategies/<int:strategy_id>/deploy', methods=['POST'])
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

@app.route('/api/me/paper/account', methods=['GET'])
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


# --- Advanced Backtesting Endpoints ---

@app.route('/api/strategies/<int:strategy_id>/walkforward', methods=['POST'])
@token_required
def walk_forward_strategy(current_user, strategy_id):
    """
    Runs a walk-forward analysis for a given strategy.
    """
    from walk_forward_analyzer import run_walk_forward_analysis

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

        results = run_walk_forward_analysis(strategy_id, optimization_params, in_sample_days, out_of_sample_days)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': 'An error occurred during walk-forward analysis.', 'details': str(e)}), 500

@app.route('/api/strategies/<int:strategy_id>/optimize', methods=['POST'])
@token_required
def optimize_strategy(current_user, strategy_id):
    """
    Runs parameter optimization for a given strategy.
    """
    from optimizer import run_optimization # Moved import to prevent circular dependency

    strategy = Strategy.query.get_or_404(strategy_id)
    if strategy.author != current_user:
        return jsonify({'error': 'You can only optimize your own strategies.'}), 403

    data = request.get_json()
    required_fields = ['parameter_name', 'start', 'end', 'step']
    if not data or not all(field in data for field in required_fields):
        return jsonify({'error': f'Missing one or more required fields: {required_fields}'}), 400

    try:
        results = run_optimization(strategy_id, data)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': 'An error occurred during optimization.', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
