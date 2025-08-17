import os
from flask import Flask, render_template
from flask_migrate import Migrate
from dotenv import load_dotenv
from flasgger import Swagger

from .extensions import db
from .routes.auth import auth_bp
from .routes.strategies import strategies_bp
from .routes.user import user_bp
from .routes.portfolios import portfolios_bp
from .routes.analysis import analysis_bp
from .routes.payments import payments_bp
from .errors import register_error_handlers

def create_app():
    """
    Application factory function.
    """
    load_dotenv() # Load environment variables from .env file

    app = Flask(__name__)

    # --- Configuration ---
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SWAGGER'] = {
        'title': 'Quant Trading Platform API',
        'uiversion': 3
    }

    # --- Extension Initialization ---
    db.init_app(app)
    Migrate(app, db)
    Swagger(app)

    # --- Blueprint Registration ---
    app.register_blueprint(auth_bp)
    app.register_blueprint(strategies_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(portfolios_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(payments_bp)

    # --- Error Handler Registration ---
    register_error_handlers(app)

    @app.route('/')
    def index():
        return render_template('index.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
