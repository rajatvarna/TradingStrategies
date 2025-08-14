import os
from flask import Flask
from flask_migrate import Migrate
from dotenv import load_dotenv
from flasgger import Swagger

from .extensions import db
from .routes.auth import auth_bp
from .routes.strategies import strategies_bp
from .routes.user import user_bp
from .routes.portfolios import portfolios_bp
from .routes.analysis import analysis_bp

def create_app():
    """
    Application factory function.
    """
    load_dotenv() # Load environment variables from .env file

    app = Flask(__name__)

    # --- Configuration ---
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///' + os.path.join(basedir, 'app.db'))
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-default-fallback-key-for-dev')
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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
