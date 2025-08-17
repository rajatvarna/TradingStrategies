from flask import jsonify

class APIError(Exception):
    """Base class for all API errors."""
    status_code = 500
    message = "An unexpected error occurred."

    def __init__(self, message=None, status_code=None, payload=None):
        super().__init__()
        if message is not None:
            self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

class ValidationError(APIError):
    """Indicates a validation error."""
    status_code = 400
    message = "Validation error."

class AuthenticationError(APIError):
    """Indicates an authentication error."""
    status_code = 401
    message = "Authentication error."

class ForbiddenError(APIError):
    """Indicates a permission error."""
    status_code = 403
    message = "You don't have permission to do that."

def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
