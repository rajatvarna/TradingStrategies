import json

def get_auth_token(client, username, password):
    """
    Helper function to get an authentication token for a user.
    """
    response = client.post('/api/login',
                           data=json.dumps({'username': username, 'password': password}),
                           content_type='application/json')
    return response.get_json()['token']
