# Database setup (local)

1. Create the database (MySQL example):

```sql
CREATE DATABASE IF NOT EXISTS stress_db;
CREATE USER IF NOT EXISTS 'stress_user'@'localhost' IDENTIFIED BY 'StrongPassword123';
GRANT ALL PRIVILEGES ON stress_db.* TO 'stress_user'@'localhost';
FLUSH PRIVILEGES;
```

2. Copy `.env.example` to `.env` and set your `DATABASE_URL`. Examples:

- MySQL (mysqlconnector or pymysql):

```
DATABASE_URL=mysql+pymysql://stress_user:StrongPassword123@localhost:3306/stress_db
```

- SQLite (dev):

```
DATABASE_URL=sqlite:///./app.db
```

3. Install dependencies and run migrations (Windows):

```
scripts\setup_local_env.bat
```

4. If you prefer, create tables quickly without migrations:

```
set FLASK_APP=app.py
python create_db.py
```

5. Run the app:

```
set FLASK_APP=app.py
flask run
```

Security notes:

- Do not commit `.env` to source control.
- Use strong passwords and limit DB user privileges.
- In production, enable SSL and use a managed DB service if possible.

---

**Note: SQLTools / VS Code may show a `mysql.user` permission warning** ⚠️

- You might see an error like:

```
SELECT user, host FROM mysql.user;
-- Execute fail: SELECT command denied to user 'stress_user'@'localhost' for table 'user'
```

- This is SQLTools trying to read the MySQL _system_ table `mysql.user` (server-level metadata). Only root/admin users can read that table, so the denial is expected and safe — **it does not mean your app failed to insert or read application data**.

**Important MySQL host note**

- MySQL treats `'user'@'localhost'` and `'user'@'127.0.0.1'` as different accounts. If your `DATABASE_URL` uses `127.0.0.1` but you created/granted privileges for `'stress_user'@'localhost'`, the app may connect but lack the necessary privileges to read/write tables.

**Fix** (run as root):

```sql
-- Check current users for stress_user
SELECT user, host FROM mysql.user WHERE user='stress_user';

-- Create/grant for 127.0.0.1 if missing
CREATE USER IF NOT EXISTS 'stress_user'@'127.0.0.1' IDENTIFIED BY 'StrongPassword123';
GRANT ALL PRIVILEGES ON stress_db.* TO 'stress_user'@'127.0.0.1';
FLUSH PRIVILEGES;

-- Ensure localhost also has grants (optional but recommended)
GRANT ALL PRIVILEGES ON stress_db.* TO 'stress_user'@'localhost';
FLUSH PRIVILEGES;
```

- Alternative: change your `DATABASE_URL` to use `localhost` instead of `127.0.0.1` to match existing grants:

```
DATABASE_URL=mysql+pymysql://stress_user:StrongPassword123@localhost:3306/stress_db
```

This should resolve the confusing SQLTools error and the "data not saving" symptom when it's caused by mismatched host accounts.

---

**SQLTools (VS Code) — set up a single MySQL connection**

1. Open the SQLTools extension panel in VS Code.
2. Remove any old connections that use `127.0.0.1`, `root`, or SQLite.
3. Create a new MySQL connection using these exact settings:
   - Driver: MySQL
   - Host: `localhost`
   - Port: `3306`
   - User: `stress_user`
   - Password: `StrongPassword123`
   - Database: `stress_db`
4. Test the connection and save it.

This ensures the editor's DB view matches the DB your Flask app uses.
