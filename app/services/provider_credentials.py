"""Encrypted provider credential overrides; environment keys remain defaults."""
import base64
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from cryptography.fernet import Fernet
from app.core.config import DirectoryConfig

NAMES = ('AIAPIROUTE_API_KEY', 'FAL_API_KEY')


def path():
    return Path(DirectoryConfig.TASKS_DIR) / '.provider-credentials.sqlite3'


def cipher():
    secret = os.getenv('SHOPIFY_API_SECRET') or os.getenv('SHOPIFY_CLIENT_SECRET') or os.getenv('ADMIN_SESSION_SECRET')
    if not secret:
        raise ValueError('Credential encryption is not configured')
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(('provider-credentials:' + secret).encode()).digest()))


@contextmanager
def database():
    target = path(); target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=10)
    try:
        db.execute('CREATE TABLE IF NOT EXISTS credentials (name TEXT PRIMARY KEY, encrypted TEXT NOT NULL)')
        yield db
        db.commit()
    finally:
        db.close()


def resolve(name, fallback=None):
    if name not in NAMES:
        raise ValueError('Unsupported provider')
    if not path().exists():
        return fallback
    with database() as db:
        row = db.execute('SELECT encrypted FROM credentials WHERE name=?', (name,)).fetchone()
    return cipher().decrypt(row[0].encode()).decode() if row else fallback


def status():
    from app.core.config import APIConfig
    return {name: {'configured': bool(getattr(APIConfig, name))} for name in NAMES}


def save(values):
    if not isinstance(values, dict) or set(values) - set(NAMES):
        raise ValueError('Invalid credential fields')
    updates = {}
    for name, value in values.items():
        if not isinstance(value, str) or len(value) > 4096:
            raise ValueError('Invalid credential')
        value = value.strip()
        if not value:
            continue
        if not value.isascii() or any(c.isspace() or ord(c) < 33 for c in value):
            raise ValueError('Invalid credential')
        updates[name] = value
    if updates:
        encrypted = [(name, cipher().encrypt(value.encode()).decode()) for name, value in updates.items()]
        with database() as db:
            db.executemany('INSERT OR REPLACE INTO credentials VALUES (?,?)', encrypted)
