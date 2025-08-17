import json
import bcrypt
from .extensions import db

# --- Database Models ---

portfolio_strategies = db.Table('portfolio_strategies',
    db.Column('portfolio_id', db.Integer, db.ForeignKey('portfolio.id'), primary_key=True),
    db.Column('strategy_id', db.Integer, db.ForeignKey('strategy.id'), primary_key=True)
)

class User(db.Model):
    """
    Represents a user of the application.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    plan_id = db.Column(db.Integer, db.ForeignKey('plan.id'), nullable=True)

    strategies = db.relationship('Strategy', backref='author', lazy='dynamic')
    api_keys = db.relationship('APIKey', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    paper_account = db.relationship('PaperAccount', backref='user', uselist=False, cascade="all, delete-orphan")
    portfolios = db.relationship('Portfolio', backref='author', lazy='dynamic')

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
    last_used = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(32), default='active', nullable=False) # active, revoked

    def to_dict(self):
        return {
            'id': self.id,
            'prefix': self.prefix,
            'created_at': self.created_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'status': self.status
        }

class Strategy(db.Model):
    """
    Represents a trading strategy stored in the database.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(64), nullable=True)
    is_public = db.Column(db.Boolean, default=False, nullable=False)
    is_paper_deployed = db.Column(db.Boolean, default=False, nullable=False)
    # The configuration for the CustomStrategy, stored as a JSON string
    config_json = db.Column(db.Text, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    backtest_result = db.relationship('BacktestResult', backref='strategy', uselist=False, cascade="all, delete-orphan")
    portfolios = db.relationship('Portfolio', secondary=portfolio_strategies, lazy='subquery',
        backref=db.backref('strategies', lazy=True))

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

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'user_id': self.user_id,
            'strategy_ids': [s.id for s in self.strategies]
        }

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

class Transaction(db.Model):
    """
    Represents a financial transaction, such as a subscription payment or a donation.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    transaction_type = db.Column(db.String(32), nullable=False)  # 'subscription' or 'donation'
    amount = db.Column(db.Float, nullable=False)
    stripe_charge_id = db.Column(db.String(128), unique=True, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

    user = db.relationship('User', backref=db.backref('transactions', lazy='dynamic'))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'transaction_type': self.transaction_type,
            'amount': self.amount,
            'stripe_charge_id': self.stripe_charge_id,
            'timestamp': self.timestamp.isoformat()
        }

class Plan(db.Model):
    """
    Represents a subscription plan.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    price = db.Column(db.Float, nullable=False)
    private_strategies_limit = db.Column(db.Integer, nullable=False)
    api_access = db.Column(db.Boolean, default=False, nullable=False)

    users = db.relationship('User', backref='plan', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'price': self.price,
            'private_strategies_limit': self.private_strategies_limit,
            'api_access': self.api_access
        }
