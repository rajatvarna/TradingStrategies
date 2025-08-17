from flask import Blueprint, request, jsonify
from web_app.models import User, db
from web_app.errors import ValidationError, AuthenticationError
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from flask import current_app

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/api/register', methods=['POST'])
def register():
    """
    Register a new user
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - username
            - email
            - password
          properties:
            username:
              type: string
              description: The user's username.
            email:
              type: string
              description: The user's email address.
            password:
              type: string
              description: The user's password.
    responses:
      201:
        description: User registered successfully
      400:
        description: Bad request (e.g., missing fields, user already exists)
    """
    data = request.get_json()
    if not data or not 'username' in data or not 'password' in data or not 'email' in data:
        raise ValidationError('Missing username, email, or password')

    if User.query.filter_by(username=data['username']).first():
        raise ValidationError('Username already exists')

    if User.query.filter_by(email=data['email']).first():
        raise ValidationError('Email address already in use')

    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/api/login', methods=['POST'])
def login():
    """
    Authenticate a user and get a JWT token
    ---
    tags:
      - Authentication
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
              description: The user's username.
            password:
              type: string
              description: The user's password.
    responses:
      200:
        description: Login successful, token returned
        schema:
          type: object
          properties:
            token:
              type: string
              description: The JWT token for authentication.
      400:
        description: Bad request (e.g., missing fields)
      401:
        description: Invalid username or password
    """
    data = request.get_json()
    if not data or not 'username' in data or not 'password' in data:
        raise ValidationError('Missing username or password')

    user = User.query.filter_by(username=data['username']).first()

    if not user or not user.check_password(data['password']):
        raise AuthenticationError('Invalid username or password')

    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.now(timezone.utc) + timedelta(hours=24)
    }, current_app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({'token': token})
