#!/usr/bin/env python3
"""Create MySQL database and user for the project.

Usage:
  # will prompt for root password if MYSQL_ROOT_PW not set
  python scripts/setup_db.py

This script connects as MySQL root and runs the SQL to create the
`stress_db` database and the `stress_user`@'localhost' user with the
password `StrongPassword123`.

It is safe to run multiple times (uses IF NOT EXISTS).
"""
import os
import sys
import getpass

try:
    import pymysql
except Exception as e:
    print("pymysql is required. Run: pip install pymysql")
    raise

ROOT_USER = os.environ.get("MYSQL_ROOT_USER", "root")
ROOT_PW = os.environ.get("MYSQL_ROOT_PW")
if not ROOT_PW:
    ROOT_PW = getpass.getpass(f"Enter MySQL root password for {ROOT_USER}: ")

DB_NAME = "stress_db"
DB_USER = "stress_user"
DB_PASS = "StrongPassword123"
DB_HOST = "localhost"
DB_PORT = 3306

SQL = f"""
CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '{DB_USER}'@'{DB_HOST}' IDENTIFIED BY '{DB_PASS}';
GRANT ALL PRIVILEGES ON `{DB_NAME}`.* TO '{DB_USER}'@'{DB_HOST}';
FLUSH PRIVILEGES;
"""

print("Connecting to MySQL as root to create database and user...")
try:
    conn = pymysql.connect(host=DB_HOST, user=ROOT_USER, password=ROOT_PW, port=DB_PORT, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cur:
        for stmt in SQL.strip().split(";\n"):
            stmt = stmt.strip()
            if not stmt:
                continue
            print("Executing:", stmt if len(stmt) < 120 else stmt[:120] + "...")
            cur.execute(stmt)
        conn.commit()

        cur.execute("SELECT user, host FROM mysql.user WHERE user=%s", (DB_USER,))
        rows = cur.fetchall()
        print("User rows:")
        for r in rows:
            print(r)

    print("Done. Created DB and user (if they did not already exist).")
except Exception as exc:
    print("Failed to create DB/user:", exc)
    sys.exit(2)


# Quick verification: try connecting as the created app user
try:
    print(f"Verifying connection as {DB_USER}@{DB_HOST}...")
    conn2 = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, cursorclass=pymysql.cursors.DictCursor)
    with conn2.cursor() as cur:
        cur.execute("SHOW TABLES;")
        print("SHOW TABLES result (empty if no tables yet):", cur.fetchall())
    conn2.close()
    print("Verification successful.")
except Exception as exc:
    print("Verification failed (this is okay if you haven't created tables yet):", exc)
    # don't fail
    pass
