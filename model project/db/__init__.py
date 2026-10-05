from pathlib import Path
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

# Shared extension objects
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = "auth.login"


def init_app(app):
    """Initialize DB, Migrate and LoginManager on the Flask app.

    Configuration values are read from environment variables with sensible
    defaults (SQLite file in the project directory when DATABASE_URL is unset).
    """
    # sensible defaults
    # Ensure SECRET_KEY is set (setdefault won't override an existing None)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or "dev-secret-change-me"
    base_dir = Path(__file__).resolve().parents[1]
    # Prefer explicit DATABASE_URL. For local development default to a MySQL URL constructed
    # from DB_USER/DB_PASS/DB_HOST/DB_NAME environment variables (safer than embedding secrets in code).
    env_db = os.environ.get("DATABASE_URL")
    if env_db:
        app.config.setdefault("SQLALCHEMY_DATABASE_URI", env_db)
    else:
        db_user = os.environ.get("DB_USER", "stress_user")
        db_pass = os.environ.get("DB_PASS", "change-me")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_name = os.environ.get("DB_NAME", "stress_db")
        default_mysql = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{db_name}"
        app.config.setdefault("SQLALCHEMY_DATABASE_URI", default_mysql)
        if db_pass == "change-me":
            import logging as _logging
            _logging.warning("Using default DB credentials for development. Set DB_PASS in .env to a secure password before production.")
    app.config.setdefault("SQLALCHEMY_TRACK_MODIFICATIONS", False)

    # Validate DB URI: enforce MySQL, localhost, and forbid SQLite / 127.0.0.1 usage
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if not db_uri:
        raise RuntimeError("No database configured. Set DATABASE_URL or DB_* environment variables to point to your MySQL instance.")
    # Disallow sqlite
    if db_uri.startswith("sqlite:"):
        raise RuntimeError("SQLite is not permitted. Configure MySQL via DATABASE_URL or DB_* environment variables.")
    # Ensure MySQL driver
    if not db_uri.startswith("mysql+"):
        raise RuntimeError("Only MySQL (mysql+pymysql) is allowed. Please set DATABASE_URL accordingly.")
    # Parse host and ensure it's localhost
    try:
        import re as _re
        m = _re.search(r"@([^:/]+)(?::\d+)?/", db_uri)
        host = m.group(1) if m else None
        if host is None:
            raise RuntimeError("Could not parse DB host from DATABASE_URL; ensure it contains host and database name.")
        if host == "127.0.0.1":
            raise RuntimeError("Using 127.0.0.1 is disallowed; use 'localhost' in DATABASE_URL or DB_HOST.")
        if host != "localhost":
            # Warn but allow non-localhost hosts (user explicitly wanted 'localhost', so enforce)
            raise RuntimeError("DB host must be 'localhost' for this configuration. Found: %s" % host)
    except Exception as _exc:
        # Re-raise with helpful message
        raise

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
