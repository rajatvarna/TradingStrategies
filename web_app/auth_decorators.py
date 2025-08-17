from functools import wraps
from flask import request, jsonify, current_app
import jwt
import bcrypt
from web_app.models import User, APIKey, db

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1] # Bearer <token>

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401

        return f(current_user, *args, **kwargs)
    return decorated

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'message': 'API key is missing!'}), 401

        # Find key by prefix
        prefix = api_key[:8]
        key_record = APIKey.query.filter_by(prefix=prefix, status='active').first()

        if key_record and bcrypt.checkpw(api_key.encode('utf-8'), key_record.key_hash.encode('utf-8')):
            # Update last_used timestamp
            from datetime import datetime
            key_record.last_used = datetime.utcnow()
            db.session.commit()

            current_user = User.query.get(key_record.user_id)
            return f(current_user, *args, **kwargs)

        return jsonify({'message': 'API key is invalid or revoked!'}), 401
    return decorated
