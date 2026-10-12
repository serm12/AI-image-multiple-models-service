"""Persist the editable original-image filename factor on the task volume."""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from app.core.config import AlgorithmConfig, DirectoryConfig


def path():
    return Path(DirectoryConfig.TASKS_DIR) / '.algorithm-settings.json'


def factor():
    try:
        value = json.loads(path().read_text('utf-8'))['factor']
        if type(value) is int and 1 <= value <= 999999999:
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return AlgorithmConfig.ALGORITHM_FACTOR


def save(value):
    if type(value) is not int or not 1 <= value <= 999999999:
        raise ValueError('Invalid algorithm factor')
    target = path(); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps({'factor': value}), 'utf-8')
    os.replace(temporary, target)


def response():
    value = factor(); year = datetime.now().year
    return {'factor': value, 'year': year, 'algorithm_value': year * value}
