import os

class TestingConfig:
    """Configuration for testing."""
    TESTING = True
    SECRET_KEY = 'a-test-secret-key'
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
