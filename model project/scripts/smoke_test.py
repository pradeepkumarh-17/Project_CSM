#!/usr/bin/env python3
"""Smoke test: register -> login -> create assessment -> verify DB rows."""
import time
import requests
import pymysql
from urllib.parse import urljoin

BASE = "http://localhost:5000"
REGISTER = urljoin(BASE, "/api/register")
LOGIN = urljoin(BASE, "/api/login")
ASSESS = urljoin(BASE, "/api/assessment")

TEST_EMAIL = "test_user@example.com"
TEST_PASS = "TestPass123"

print("Starting smoke test. Ensure the app is running on http://localhost:5000")

s = requests.Session()

# Try to register (ignore if exists)
r = s.post(REGISTER, json={"email": TEST_EMAIL, "password": TEST_PASS, "name": "Tester"})
print("Register", r.status_code, r.text)

# Login
r = s.post(LOGIN, json={"email": TEST_EMAIL, "password": TEST_PASS})
print("Login", r.status_code, r.text)
if r.status_code != 200:
    print("Login failed; aborting smoke test.")
    raise SystemExit(2)

# Create assessment
r = s.post(ASSESS, json={"stress_level": "High", "assessment_data": {"source": "smoke_test"}})
print("Assessment create", r.status_code, r.text)

# Wait a moment for DB writes
time.sleep(1)

# Verify DB (connect as stress_user)
DB_HOST = "localhost"
DB_USER = "stress_user"
DB_PASS = "StrongPassword123"
DB_NAME = "stress_db"
DB_PORT = 3306

try:
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, db=DB_NAME, port=DB_PORT, cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cur:
        cur.execute("SELECT id, email, created_at FROM `user` WHERE email=%s", (TEST_EMAIL,))
        usr = cur.fetchall()
        print("User rows:", usr)
        cur.execute("SELECT id, user_id, stress_level, created_at FROM `assessment` ORDER BY id DESC LIMIT 5")
        asses = cur.fetchall()
        print("Recent assessments:", asses)
    conn.close()
except Exception as exc:
    print("DB verification failed:", exc)
    raise

print("Smoke test completed.")
