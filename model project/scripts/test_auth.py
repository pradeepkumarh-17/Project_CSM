# Test register and login using Flask test client
import sys
from pathlib import Path
# Ensure project root is on sys.path (handles spaces in path)
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from app import app
from db import db
import json

with app.app_context():
    print('DB URI:', app.config.get('SQLALCHEMY_DATABASE_URI'))
    print('Attempting to create all tables (no-op if exist)')
    db.create_all()

    client = app.test_client()

    # Test register
    reg_data = {'email': 'tester@example.com', 'password': 'TestPass123!'}
    r = client.post('/api/register', data=json.dumps(reg_data), content_type='application/json')
    print('/api/register', r.status_code, r.get_data(as_text=True))

    # Test login
    r2 = client.post('/api/login', data=json.dumps(reg_data), content_type='application/json')
    print('/api/login', r2.status_code, r2.get_data(as_text=True))

    # Test /api/me with cookies
    # Flask test client stores cookies automatically when using the same client
    r3 = client.get('/api/me')
    print('/api/me', r3.status_code, r3.get_data(as_text=True))
