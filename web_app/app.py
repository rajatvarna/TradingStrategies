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

    def to_dict(self):
        """
        Serializes the Strategy object to a dictionary.
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'is_public': self.is_public,
            'config': json.loads(self.config_json)
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


if __name__ == '__main__':
    app.run(debug=True)
