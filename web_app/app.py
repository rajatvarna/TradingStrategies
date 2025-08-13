import os
import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone

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

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def __repr__(self):
        return f'<User {self.username}>'

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


from functools import wraps

# --- Auth Decorator ---
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


if __name__ == '__main__':
    app.run(debug=True)
