import sys
from pathlib import Path
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from app import app
from db import db
from sqlalchemy import inspect

with app.app_context():
    uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    print('SQLALCHEMY_DATABASE_URI =', uri)
    insp = inspect(db.engine)
    tables = insp.get_table_names()
    print('Tables:', tables)
    # show counts
    for t in tables:
        try:
            c = db.session.execute(f'SELECT COUNT(1) FROM "{t}"').scalar()
            print(f"{t}: {c} rows")
        except Exception as e:
            print(f"{t}: error counting rows ->", e)
