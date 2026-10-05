#!/usr/bin/env python3
"""Create a `.env` file from `.env.example` with the required DB_* variables.

This writes `.env` in the project root. It will overwrite an existing `.env` only
if you confirm.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"

content = EXAMPLE.read_text()
# Ensure DB env keys are present in .env.example; otherwise append a canonical block
if "DB_USER" not in content:
    content += "\n# DB config (recommended)\nDB_USER=stress_user\nDB_PASS=StrongPassword123\nDB_HOST=localhost\nDB_NAME=stress_db\n"

if TARGET.exists():
    ans = input(".env already exists. Overwrite? (y/N): ")
    if ans.lower() != "y":
        print("Cancelled. .env left unchanged.")
        raise SystemExit(0)

TARGET.write_text(content)
print("Wrote .env (update SECRET_KEY before production).")
