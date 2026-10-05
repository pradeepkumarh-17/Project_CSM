# Quick test: ensure DB connection works using the Flask app config
from app import app
from db import db

with app.app_context():
    try:
        r = db.session.execute('SELECT 1').scalar()
        print('DB test query returned:', r)
    except Exception as exc:
        print('DB connection test failed:', exc)
