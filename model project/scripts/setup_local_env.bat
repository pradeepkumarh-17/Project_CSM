@echo off
REM Create and activate venv, install requirements (Windows)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
REM Set FLASK_APP and run migrations
set FLASK_APP=app.py
flask db init 2>nul || echo "migrations already initialized"
flask db migrate -m "initial"
flask db upgrade
@echo on
