"""Simple helper to create DB tables without using migrations.

Usage (PowerShell):
    set FLASK_APP="app.py"; python create_db.py

This runs db.create_all() inside the app context; use Flask-Migrate for production.
"""
from app import app
from db import db

with app.app_context():
    db.create_all()
    print("Database tables created")
